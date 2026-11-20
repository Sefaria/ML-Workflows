"""Ranking metrics for roshei tevot expansion.

Each row's candidates are what inference would see: the lexicon lookup for the token, filtered by the letter check (data.attach_candidates).
Rows with fewer than two candidates are left out. A row is answerable when the spelling folds match a candidate to the gold expansion, and any spelling of it counts as correct;
ranking metrics are over answerable rows, and `end_to_end_top1` counts the rest as errors.

Scores: siamese = logit_scale * cosine, cross = the head's logit. Both get `+ λ * log(1 + lexicon frequency)` for each λ in the sweep;
λ is chosen on `val` and reused unchanged on every other split. The frequency baseline is the lexicon's most frequent candidate.
"""
from collections import defaultdict
from typing import Optional, Callable

import numpy as np

from .data import encode_rows, encode_texts, score_pairs
from .index import prior
from .vendor.spelling import same_expansion

PRIOR_WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
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
        if row.get("units"):  # sense classes: right when the candidate is the label's class
            hits = [rank for rank, i in enumerate(order) if row["units"][i] == row["target_unit"]]
            frequency_correct = row["units"][0] == row["target_unit"]
        else:
            hits = [i for i, text in enumerate(ranked) if target is not None and same_expansion(text, target)]
            frequency_correct = target is not None and same_expansion(row["candidates"][0], target)
        out.append({
            "rank": hits[0] if hits else None,
            "frequency_correct": frequency_correct,
            "confidence": top_two[0] - top_two[1] if len(top_two) > 1 else top_two[0],
        })
    return out


def sense_share_band(row: dict) -> str:
    """How common the label's sense is within its key, from the candidates' train counts: `majority` (the most-trained class),
    `common` (10% or more of the key's train rows), `rare` (under 10%), `unseen_in_train` (count 0), or `unanswerable`."""
    if row.get("target") is None:
        return "unanswerable"
    units, freqs = row.get("units"), row["freqs"]
    index = units.index(row["target_unit"]) if units else next(i for i, c in enumerate(row["candidates"]) if c == row["target"])
    total = sum(freqs)
    if freqs[index] == 0 or total == 0:
        return "unseen_in_train"
    if freqs[index] == max(freqs):
        return "majority"
    return "common" if freqs[index] / total >= 0.1 else "rare"


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


BALANCED_BANDS = ("majority", "common", "rare")


def balanced_top1(items: list[dict], rows: list[dict]) -> Optional[float]:
    """Mean top-1 over the sense-share bands (majority, common, rare) that have answerable rows, so a rare sense counts as much as a
    frequent one. Overall top-1 is dominated by majority senses and rewards a strong frequency prior; this does not."""
    bands = grouped(items, rows, sense_share_band, min_rows=1)
    values = [bands[b]["top1"] for b in BALANCED_BANDS if b in bands and "top1" in bands[b]]
    return float(np.mean(values)) if values else None


def evaluate_splits(model, tokenizer, split_rows: dict[str, list[dict]], device: str, logit_scale: float,
                    prior_weights=PRIOR_WEIGHTS, prior_select: str = "balanced", keep_scores: bool = False, **encode_settings) -> dict:
    """Score every split once, sweep λ, pick λ on val (or the first split), and report each split at every λ plus details at the chosen λ.
    `prior_select`: "balanced" picks the λ with the best balanced_top1 (ties go to the smaller λ); "top1" the best overall top-1.
    `keep_scores` adds the raw candidate scores per split under "scores" (for combining models offline)."""
    if prior_select not in ("balanced", "top1"):
        raise ValueError(f"prior_select must be 'balanced' or 'top1', not {prior_select!r}")
    scores = {split: model_scores(model, tokenizer, rows, device, logit_scale, **encode_settings) for split, rows in split_rows.items()}
    sweep = {}
    for split, rows in split_rows.items():
        sweep[split] = {}
        for w in prior_weights:
            items = outcomes(rows, scores[split], w)
            sweep[split][str(w)] = {**summarise(items), "balanced_top1": balanced_top1(items, rows)}
    selection_split = "val" if "val" in split_rows else next(iter(split_rows))
    metric = "balanced_top1" if prior_select == "balanced" else "top1"
    chosen = max(sorted(prior_weights), key=lambda w: sweep[selection_split][str(w)].get(metric) or 0.0)
    report = {"prior_weight": chosen, "prior_selected_on": selection_split, "prior_select": prior_select, "splits": {}}
    for split, rows in split_rows.items():
        items = outcomes(rows, scores[split], chosen)
        report["splits"][split] = {
            "overall": {**summarise(items), "balanced_top1": balanced_top1(items, rows)},
            "prior_sweep": sweep[split],
            "by_source": grouped(items, rows, lambda r: r["source"]),
            "by_kind": grouped(items, rows, lambda r: r.get("kind") or "abbreviation"),
            "by_sense_share": grouped(items, rows, sense_share_band, min_rows=1),
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
    if keep_scores:
        report["scores"] = scores
    return report


def score_records(split_rows: dict[str, list[dict]], scores_by_model: dict[str, dict[str, list[list[float]]]]) -> list[dict]:
    """One record per row with every model's raw candidate scores, for combining models offline (rows are in the same order for every model)."""
    out = []
    for split, rows in split_rows.items():
        for i, row in enumerate(rows):
            out.append({"split": split, "id": row.get("id"), "key": row.get("key"), "source": row.get("source"), "kind": row.get("kind"),
                        "category": row.get("category"), "band": sense_share_band(row), "candidates": row["candidates"], "units": row.get("units"),
                        "target_unit": row.get("target_unit"), "target": row.get("target"), "freqs": row["freqs"],
                        "scores": {label: by_split[split][i] for label, by_split in scores_by_model.items()}})
    return out


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
