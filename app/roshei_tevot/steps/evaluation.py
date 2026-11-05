"""Ranking metrics for roshei tevot expansion.

Each row's candidates are what inference would see: the lexicon lookup for the token, filtered by the letter check (data.attach_candidates).
Rows with fewer than two candidates are left out. A row is answerable when the spelling folds match a candidate to the gold expansion, and any spelling of it counts as correct;
ranking metrics are over answerable rows, and `end_to_end_top1` counts the rest as errors.

Scores: siamese = logit_scale * cosine, cross = the head's logit. Both get `+ λ * log(1 + lexicon frequency)` for each λ in the sweep;
λ is chosen on `val` and reused unchanged on every other split. The frequency baseline is the lexicon's most frequent candidate.
"""
from collections import defaultdict
from typing import Callable

import numpy as np

from .data import encode_rows, encode_texts, score_pairs
from .index import prior
from .vendor.spelling import same_expansion

PRIOR_WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0)
COVERAGE_LEVELS = (0.5, 0.8, 0.9)
SUCCESS_SPLITS = ("val", "test_seen_keys")
SUCCESS_MARGIN = 0.03


def model_scores(model, tokenizer, rows: list[dict], device: str, logit_scale: float, **encode_settings) -> list[list[float]]:
    """Raw model score for every candidate of every row, in the row's candidate order."""
    if getattr(model, "architecture", "siamese") == "cross":
        return score_pairs(model, tokenizer, rows, device, **encode_settings)
    queries = encode_rows(model, tokenizer, rows, device, **encode_settings)
    texts = sorted({t for r in rows for t in r["candidates"]})
    embeddings = encode_texts(model, tokenizer, texts, device)
    position = {t: i for i, t in enumerate(texts)}
    return [(logit_scale * (embeddings[[position[t] for t in r["candidates"]]] @ q)).tolist() for r, q in zip(rows, queries)]


def outcomes(rows: list[dict], scores: list[list[float]], prior_weight: float) -> list[dict]:
    out = []
    for row, raw in zip(rows, scores):
        combined = [s + prior_weight * prior(f) for s, f in zip(raw, row["freqs"])]
        order = sorted(range(len(combined)), key=lambda i: -combined[i])
        ranked = [row["candidates"][i] for i in order]
        top_two = [combined[i] for i in order[:2]]
        target = row["target"]
        hits = [i for i, text in enumerate(ranked) if target is not None and same_expansion(text, target)]
        out.append({
            "rank": hits[0] if hits else None,
            "frequency_correct": target is not None and same_expansion(row["candidates"][0], target),
            "confidence": top_two[0] - top_two[1] if len(top_two) > 1 else top_two[0],
        })
    return out


def summarise(items: list[dict]) -> dict:
    total = len(items)
    answerable = [o for o in items if o["rank"] is not None]
    if not answerable:
        return {"rows": total, "answerable": 0}
    ranks = np.array([o["rank"] for o in answerable])
    top1 = ranks == 0
    metrics = {
        "rows": total,
        "answerable": len(answerable),
        "answerable_fraction": len(answerable) / total,
        "top1": float(top1.mean()),
        "top3": float((ranks < 3).mean()),
        "mrr": float((1.0 / (ranks + 1)).mean()),
        "end_to_end_top1": float(top1.sum() / total),
        "frequency_top1": float(np.mean([o["frequency_correct"] for o in answerable])),
    }
    # Accuracy among the most confident answers: what the model would get right if it abstained on the rest.
    order = np.argsort(-np.array([o["confidence"] for o in answerable]))
    for level in COVERAGE_LEVELS:
        kept = order[: max(1, int(round(level * len(order))))]
        metrics[f"top1_at_coverage_{int(level * 100)}"] = float(top1[kept].mean())
    return metrics


def grouped(items: list[dict], rows: list[dict], key: Callable[[dict], str], min_rows: int = 50) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for item, row in zip(items, rows):
        groups[key(row)].append(item)
    return {name: summarise(members) for name, members in sorted(groups.items()) if len(members) >= min_rows}


def evaluate_splits(model, tokenizer, split_rows: dict[str, list[dict]], device: str, logit_scale: float,
                    prior_weights=PRIOR_WEIGHTS, **encode_settings) -> dict:
    """Score every split once, sweep λ, pick λ on val (or the first split), and report each split at every λ plus details at the chosen λ."""
    scores = {split: model_scores(model, tokenizer, rows, device, logit_scale, **encode_settings) for split, rows in split_rows.items()}
    sweep = {split: {str(w): summarise(outcomes(split_rows[split], scores[split], w)) for w in prior_weights} for split in split_rows}
    selection_split = "val" if "val" in split_rows else next(iter(split_rows))
    chosen = max(prior_weights, key=lambda w: sweep[selection_split][str(w)].get("top1", 0.0))
    report = {"prior_weight": chosen, "prior_selected_on": selection_split, "splits": {}}
    for split, rows in split_rows.items():
        items = outcomes(rows, scores[split], chosen)
        report["splits"][split] = {
            "overall": summarise(items),
            "prior_sweep": sweep[split],
            "by_source": grouped(items, rows, lambda r: r["source"]),
            "by_category": grouped(items, rows, lambda r: r["category"] or ""),
        }
    checks = {}
    for split in SUCCESS_SPLITS:
        if split in report["splits"] and "top1" in report["splits"][split]["overall"]:
            o = report["splits"][split]["overall"]
            checks[split] = {"top1": o["top1"], "frequency_top1": o["frequency_top1"], "gain": o["top1"] - o["frequency_top1"],
                             "passes": o["top1"] - o["frequency_top1"] >= SUCCESS_MARGIN}
    report["success"] = {"margin": SUCCESS_MARGIN, "checks": checks,
                         "passes": bool(checks) and len(checks) == len(SUCCESS_SPLITS) and all(c["passes"] for c in checks.values())}
    return report


def quick_top1(model, tokenizer, rows: list[dict], device: str, **encode_settings) -> dict:
    """Validation top-1 at λ = 0 during training (answerable rows), with the frequency baseline for scale."""
    was_training = model.training
    model.eval()
    try:
        scale = float(model.scale()) if hasattr(model, "scale") and callable(getattr(model, "scale")) else 1.0
        summary = summarise(outcomes(rows, model_scores(model, tokenizer, rows, device, scale, **encode_settings), 0.0))
    finally:
        model.train(was_training)
    return {"top1": summary.get("top1", 0.0), "frequency_top1": summary.get("frequency_top1", 0.0), "mrr": summary.get("mrr", 0.0)}
