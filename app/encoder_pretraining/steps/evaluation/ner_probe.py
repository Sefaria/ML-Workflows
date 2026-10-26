import bisect
import re
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

from ..attention import resolve_attention_implementation


ENTITY_LABELS = ("Person", "Citation")
TAGS = ["O"] + [f"{prefix}-{label}" for label in ENTITY_LABELS for prefix in ("B", "I")]
TAG_TO_ID = {tag: i for i, tag in enumerate(TAGS)}
WORD_PATTERN = re.compile(r"\S+")


def word_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in WORD_PATTERN.finditer(text)]


def gold_word_tags(text: str, spans: list[dict]) -> list[str]:
    """BIO tag per whitespace word. A word belongs to a span if it overlaps it."""
    words = word_spans(text)
    tags = ["O"] * len(words)
    for span in sorted(spans, key=lambda s: s["start"]):
        inside = [i for i, (start, end) in enumerate(words) if start < span["end"] and end > span["start"]]
        for position, word_index in enumerate(inside):
            tags[word_index] = f"{'B' if position == 0 else 'I'}-{span['label']}"
    return tags


def tags_to_spans(tags: list[str]) -> set[tuple[int, int, str]]:
    spans, start, label = set(), None, None
    for index, tag in enumerate(tags + ["O"]):
        prefix, _, tag_label = tag.partition("-")
        continues = prefix == "I" and label == tag_label and start is not None
        if start is not None and not continues:
            spans.add((start, index - 1, label))
            start, label = None, None
        if prefix == "B" or (prefix == "I" and start is None):
            start, label = index, tag_label
    return spans


def span_scores(gold: list[set], predicted: list[set]) -> dict:
    """Exact-match span precision, recall and F1, micro-averaged and per label."""
    def prf(g: int, p: int, tp: int) -> dict:
        precision = tp / p if p else 0.0
        recall = tp / g if g else 0.0
        return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}

    scores = {}
    for name, keep in [("micro", None), *[(label, label) for label in ENTITY_LABELS]]:
        g = sum(1 for spans in gold for s in spans if keep is None or s[2] == keep)
        p = sum(1 for spans in predicted for s in spans if keep is None or s[2] == keep)
        tp = sum(1 for gs, ps in zip(gold, predicted) for s in gs & ps if keep is None or s[2] == keep)
        scores[name] = prf(g, p, tp)
    return scores


def encode_examples(tokenizer, examples: list[dict], max_length: int) -> list[dict]:
    """Token ids plus a label on the first token of every word (the other tokens are ignored)."""
    encoded = []
    for example in examples:
        text = example["text"]
        words = word_spans(text)
        starts = [start for start, _ in words]
        tags = gold_word_tags(text, example["spans"])
        enc = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=max_length)
        labels, first_token_of_word = [], {}
        for token_index, (start, end) in enumerate(enc["offset_mapping"]):
            # Offsets include a word's leading space, so find the word by the token's last character.
            last = end - 1
            word_index = bisect.bisect_right(starts, last) - 1
            if end == start or word_index < 0 or word_index in first_token_of_word or last >= words[word_index][1]:
                labels.append(-100)
            else:
                first_token_of_word[word_index] = token_index
                labels.append(TAG_TO_ID[tags[word_index]])
        encoded.append({"input_ids": enc["input_ids"], "labels": labels, "word_tokens": first_token_of_word, "tags": tags})
    return encoded


def _batches(items: list[dict], batch_size: int, pad_id: int, shuffle_generator: Optional[np.random.Generator]):
    order = np.arange(len(items))
    if shuffle_generator is not None:
        shuffle_generator.shuffle(order)
    for start in range(0, len(items), batch_size):
        chunk = [items[i] for i in order[start : start + batch_size]]
        width = max(len(item["input_ids"]) for item in chunk)
        input_ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        attention = torch.zeros((len(chunk), width), dtype=torch.long)
        labels = torch.full((len(chunk), width), -100, dtype=torch.long)
        for row, item in enumerate(chunk):
            n = len(item["input_ids"])
            input_ids[row, :n] = torch.tensor(item["input_ids"])
            attention[row, :n] = 1
            labels[row, :n] = torch.tensor(item["labels"])
        yield chunk, input_ids, attention, labels


def fine_tune_and_score(
    model_path: str,
    tokenizer,
    train: list[dict],
    test: list[dict],
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    attn_implementation: str = "auto",
    device: Optional[str] = None,
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        num_labels=len(TAGS),
        id2label=dict(enumerate(TAGS)),
        label2id=TAG_TO_ID,
        attn_implementation=resolve_attention_implementation(attn_implementation),
        reference_compile=False,
    ).to(device)
    steps = epochs * ((len(train) + batch_size - 1) // batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(0.1 * steps)), steps)
    generator = np.random.default_rng(seed)
    pad_id = tokenizer.pad_token_id

    model.train()
    for _ in range(epochs):
        for _, input_ids, attention, labels in _batches(train, batch_size, pad_id, generator):
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = model(input_ids=input_ids.to(device), attention_mask=attention.to(device), labels=labels.to(device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    model.eval()
    gold, predicted = [], []
    with torch.no_grad():
        for chunk, input_ids, attention, _ in _batches(test, batch_size, pad_id, None):
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(input_ids=input_ids.to(device), attention_mask=attention.to(device)).logits
            best = logits.argmax(dim=-1).cpu()
            for row, item in enumerate(chunk):
                word_tags = ["O"] * len(item["tags"])
                for word_index, token_index in item["word_tokens"].items():
                    word_tags[word_index] = TAGS[int(best[row, token_index])]
                gold.append(tags_to_spans(item["tags"]))
                predicted.append(tags_to_spans(word_tags))
    del model, optimizer
    if device == "cuda":
        torch.cuda.empty_cache()
    return span_scores(gold, predicted)


def cross_validated_probe(
    model_path: str,
    examples: list[dict],
    folds: int,
    seeds: list[int],
    epochs: int,
    learning_rate: float,
    batch_size: int,
    max_length: int,
    attn_implementation: str = "auto",
) -> list[dict]:
    """K-fold fine-tune and score. Fold membership depends on the seed only, so every model sees the same splits."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    encoded = encode_examples(tokenizer, examples, max_length)
    runs = []
    for seed in seeds:
        order = np.random.default_rng(seed).permutation(len(encoded))
        for fold in range(folds):
            test_idx = set(order[fold::folds].tolist())
            train = [e for i, e in enumerate(encoded) if i not in test_idx]
            test = [e for i, e in enumerate(encoded) if i in test_idx]
            scores = fine_tune_and_score(model_path, tokenizer, train, test, epochs, learning_rate, batch_size, seed, attn_implementation)
            runs.append({"seed": seed, "fold": fold, "scores": scores})
            print(f"seed={seed} fold={fold} micro_f1={scores['micro']['f1']:.4f}")
    return runs
