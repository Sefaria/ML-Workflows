"""Ranking metrics for roshei tevot expansion, in two settings.

  lexicon  the candidates the dataset row lists for its key (what the lexicon says for that abbreviation)
  index    what ExpansionIndex returns for the token: lexicon entries for the token as written or any prefix-stripped reading,
           minus the ones the letter checker rejects (the setting the model is deployed in)

A row is answerable when its gold expansion is among the candidates; ranking metrics are over answerable rows, and `end_to_end`
counts the rest as errors. The frequency baseline is the lexicon's most frequent candidate.
"""
from collections import defaultdict
from typing import Callable, Optional

import numpy as np
import torch

from .data import encode_rows, encode_texts
from .index import ExpansionIndex

COVERAGE_LEVELS = (0.5, 0.8, 0.9)


def _confidence(ranked: list[tuple[str, float, int]]) -> float:
    """Margin between the best and second-best score; a lone candidate gets its own cosine."""
    return ranked[0][1] - ranked[1][1] if len(ranked) > 1 else ranked[0][1]


def summarise(outcomes: list[dict]) -> dict:
    """outcomes: one dict per row with `gold`, `ranked` ([(text, score, freq)] best first) and `freq_top`."""
    total = len(outcomes)
    answerable = [o for o in outcomes if any(t == o["gold"] for t, _s, _f in o["ranked"])]
    if not answerable:
        return {"rows": total, "answerable": 0}
    ranks = np.array([[t for t, _s, _f in o["ranked"]].index(o["gold"]) for o in answerable])
    top1 = np.array([o["ranked"][0][0] == o["gold"] for o in answerable])
    metrics = {
        "rows": total,
        "answerable": len(answerable),
        "answerable_fraction": len(answerable) / total,
        "top1": float(top1.mean()),
        "top3": float((ranks < 3).mean()),
        "top5": float((ranks < 5).mean()),
        "mrr": float((1.0 / (ranks + 1)).mean()),
        "end_to_end_top1": float(top1.sum() / total),
        "frequency_top1": float(np.mean([o["freq_top"] == o["gold"] for o in answerable])),
    }
    # Accuracy among the most confident answers: what the model would get right if it abstained on the rest.
    confidence = np.array([_confidence(o["ranked"]) for o in answerable])
    order = np.argsort(-confidence)
    for level in COVERAGE_LEVELS:
        kept = order[: max(1, int(round(level * len(order))))]
        metrics[f"top1_at_coverage_{int(level * 100)}"] = float(top1[kept].mean())
    return metrics


def grouped(outcomes: list[dict], rows: list[dict], key: Callable[[dict], str], min_rows: int = 50) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for outcome, row in zip(outcomes, rows):
        groups[key(row)].append(outcome)
    return {name: summarise(items) for name, items in sorted(groups.items()) if len(items) >= min_rows}


def evaluate_split(model, tokenizer, rows: list[dict], index: Optional[ExpansionIndex], device: str, scan_unknown: bool = False, **encode_settings) -> dict:
    queries = encode_rows(model, tokenizer, rows, device, **encode_settings)

    lexicon_texts = sorted({t for r in rows for t in r["candidates"]})
    lexicon_embeddings = encode_texts(model, tokenizer, lexicon_texts, device)
    position = {t: i for i, t in enumerate(lexicon_texts)}

    lexicon_outcomes, index_outcomes = [], []
    for row, query in zip(rows, queries):
        if row["candidates"]:
            scores = lexicon_embeddings[[position[t] for t in row["candidates"]]] @ query
            order = torch.argsort(scores, descending=True).tolist()
            ranked = [(row["candidates"][i], float(scores[i]), 0) for i in order]
        else:
            ranked = []
        lexicon_outcomes.append({"gold": row["base_expansion"], "ranked": ranked, "freq_top": row["candidates"][0] if row["candidates"] else None})
        if index is not None:
            found = index.rank(row["surface"], query, scan_unknown)
            freq_top = max(found, key=lambda x: x[2])[0] if found else None
            index_outcomes.append({"gold": row["base_expansion"], "ranked": found, "freq_top": freq_top})

    report = {"lexicon": {"overall": summarise(lexicon_outcomes), "by_source": grouped(lexicon_outcomes, rows, lambda r: r["source"]),
                          "by_category": grouped(lexicon_outcomes, rows, lambda r: r["category"] or "")}}
    if index is not None:
        report["index"] = {"overall": summarise(index_outcomes), "by_source": grouped(index_outcomes, rows, lambda r: r["source"]),
                           "by_category": grouped(index_outcomes, rows, lambda r: r["category"] or "")}
    return report
