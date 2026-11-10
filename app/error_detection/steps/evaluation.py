"""Word-level metrics for the error class, threshold choice, and the two baselines (lexicon, BEREL masked-LM)."""
import re
from collections import Counter, defaultdict
from typing import Optional

import numpy as np
import torch

from .data import FIX_KINDS, IGNORE, TokenCollator

LETTERS = re.compile("[^א-ת]")


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve (step-wise, as sklearn's average_precision_score)."""
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    hits = labels[order]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / positives)


def prf(predicted: np.ndarray, labels: np.ndarray) -> dict:
    tp = int((predicted & (labels == 1)).sum())
    fp = int((predicted & (labels == 0)).sum())
    fn = int((~predicted & (labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """The score threshold with the highest F1 (scores at or above it are flagged)."""
    order = np.argsort(-scores, kind="stable")
    hits = labels[order]
    tp = np.cumsum(hits)
    k = np.arange(1, len(hits) + 1)
    f1 = 2 * tp / (k + labels.sum())
    return float(scores[order][int(np.argmax(f1))])


def token_metrics(eval_pred) -> dict:
    """`compute_metrics` for the Trainer: PR-AUC and best F1 over labelled first pieces."""
    logits, labels = eval_pred
    logits, labels = np.asarray(logits), np.asarray(labels)
    mask = labels != IGNORE
    shifted = logits - logits.max(-1, keepdims=True)
    probs = np.exp(shifted)[..., 1] / np.exp(shifted).sum(-1)
    scores, gold = probs[mask], labels[mask].astype(int)
    threshold = best_threshold(scores, gold)
    return {"pr_auc": average_precision(scores, gold), "best_f1": prf(scores >= threshold, gold)["f1"], "threshold": threshold,
            "positive_rate": float(gold.mean())}


@torch.no_grad()
def predict_words(model, tokenizer, chunks: list[dict], device: str, batch_size: int = 64) -> list[dict]:
    """One record per labelled word: chunk id, word index, gold label, kind, version and the model's error probability."""
    collate = TokenCollator(tokenizer.pad_token_id)
    out = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        inputs = collate(batch)
        labels = inputs.pop("labels")
        logits = model(**{k: v.to(device) for k, v in inputs.items()}).logits.float()
        probs = torch.softmax(logits, -1)[..., 1].cpu()
        for r, chunk in enumerate(batch):
            positions = (labels[r] != IGNORE).nonzero().flatten().tolist()
            for w, p in enumerate(positions):
                out.append({"chunk": chunk["id"], "word_index": chunk["word_index"][w], "label": int(labels[r, p]),
                            "kind": chunk["word_kind"][w], "version": chunk["version"], "score": float(probs[r, p])})
    return out


@torch.no_grad()
def mlm_word_scores(mlm, tokenizer, chunks: list[dict], device: str, batch_size: int = 256) -> list[float]:
    """Baseline: for each labelled word, minus the lowest log-probability of any of its pieces when that piece is masked (higher is more
    suspicious). Word order matches `predict_words`."""
    scores = []
    for chunk in chunks:
        ids = torch.as_tensor(np.asarray(chunk["input_ids"]), dtype=torch.long)
        firsts = [i for i, label in enumerate(chunk["labels"]) if label != IGNORE]
        bounds = firsts[1:] + [len(ids) - 1]
        positions = list(range(1, len(ids) - 1))
        piece_lp = {}
        for s in range(0, len(positions), batch_size):
            ps = positions[s:s + batch_size]
            batch = ids.repeat(len(ps), 1)
            batch[torch.arange(len(ps)), torch.tensor(ps)] = tokenizer.mask_token_id
            hidden = mlm.bert(input_ids=batch.to(device)).last_hidden_state
            logits = mlm.cls(hidden[torch.arange(len(ps)), torch.tensor(ps)]).float()
            lp = torch.log_softmax(logits, -1)[torch.arange(len(ps)), ids[ps].to(device)].cpu().tolist()
            piece_lp.update(zip(ps, lp))
        for start, end in zip(firsts, bounds):
            scores.append(-min(piece_lp[p] for p in range(start, end)))
    return scores


def build_lexicon(rows: list[dict], min_count: int = 3) -> set:
    """Letter skeletons seen at least `min_count` times in corrected training text."""
    counts = Counter(LETTERS.sub("", token) for row in rows for token in row["tokens"])
    return {word for word, n in counts.items() if word and n >= min_count}


def lexicon_scores(chunks: list[dict], rows_by_id: dict, lexicon: set) -> list[float]:
    """Baseline: 1 for a word whose letter skeleton is not in the lexicon, else 0. Word order matches `predict_words`."""
    scores = []
    for chunk in chunks:
        tokens = rows_by_id[chunk["row_id"]]["tokens"]
        for i in chunk["word_index"]:
            skeleton = LETTERS.sub("", tokens[i])
            scores.append(float(bool(skeleton) and skeleton not in lexicon))
    return scores


def summarise(words: list[dict], threshold: Optional[float] = None, scores: Optional[list[float]] = None) -> dict:
    """Metrics at `threshold` (chosen here by best F1 when None) with recall by fix kind and the versions with the most errors."""
    s = np.array(scores if scores is not None else [w["score"] for w in words], dtype=float)
    y = np.array([w["label"] for w in words], dtype=int)
    chosen = best_threshold(s, y) if threshold is None else threshold
    flagged = s >= chosen
    report = {"words": len(words), "errors": int(y.sum()), "pr_auc": average_precision(s, y), "threshold": chosen, **prf(flagged, y)}
    by_kind = {}
    for kind in FIX_KINDS:
        idx = np.array([w["kind"] == kind for w in words])
        if idx.any():
            by_kind[kind] = {"errors": int(idx.sum()), "recall": float(flagged[idx].mean())}
    report["recall_by_kind"] = by_kind
    per_version = defaultdict(lambda: [0, 0, 0])
    for w, f in zip(words, flagged):
        acc = per_version[w["version"]]
        acc[0] += w["label"]; acc[1] += int(f and w["label"] == 1); acc[2] += int(f and w["label"] == 0)
    report["by_version"] = {v: {"errors": e, "found": t, "false_flags": fp} for v, (e, t, fp) in
                            sorted(per_version.items(), key=lambda kv: -kv[1][0])[:30]}
    return report


def false_positive_sample(words: list[dict], rows_by_id: dict, threshold: float, n: int = 100, context: int = 8, seed: int = 0) -> list[dict]:
    """Words flagged at `threshold` but labelled 0, with context, for hand audit (unedited errors are labelled 0)."""
    candidates = [w for w in words if w["label"] == 0 and w["score"] >= threshold]
    rng = np.random.default_rng(seed)
    picked = [candidates[i] for i in rng.choice(len(candidates), size=min(n, len(candidates)), replace=False)] if candidates else []
    out = []
    for w in picked:
        row = rows_by_id[w["chunk"].rsplit("#", 1)[0]]
        i = w["word_index"]
        out.append({"id": row["id"], "ref": row["ref"], "version": row["version"], "score": round(w["score"], 4), "word": row["tokens"][i],
                    "before": " ".join(row["tokens"][max(0, i - context):i]), "after": " ".join(row["tokens"][i + 1:i + 1 + context])})
    return out
