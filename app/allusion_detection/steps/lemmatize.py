"""Shoshan lemmatization: given tokenized text units, return the full, unfiltered
top-K lemma candidates (with cosine scores) per word.

Deliberately does NOT apply Shoshan's own POS-based candidate filtering before
ranking -- that head's guess hasn't been validated for short/rare rabbinic
word-forms, and re-deciding whether to trust it later must not require re-running
the (expensive) transformer encoding. So every word's full ranking over the entire
~117k-lemma bank is kept, with raw cosine scores: a POS filter, similarity
threshold, or cutoff K' <= K can all be applied afterward, purely offline, from the
cached output of this step.

K=10 was chosen as headroom over the K=5 found sufficient in earlier experiments
(see Sefaria/translation-engine, experiments/text_matching) to recover known
top-1 failures via disjunctive matching.
"""
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from prefect import task

TOP_K = 10


def _patch_for_topk(model, k: int = TOP_K):
    """Replace _retrieve entirely (not wrap it) so the similarity matmul against the
    full lemma bank runs once per batch, not twice. See module docstring for why this
    skips the model's own POS-based candidate filtering."""
    topk_log = []

    def patched_retrieve(qc, cands):
        sims = qc.detach().cpu().numpy() @ model.L.T  # [chunk, n_bank], computed once
        n = sims.shape[0]
        part = np.argpartition(-sims, k, axis=1)[:, :k]
        part_scores = np.take_along_axis(sims, part, axis=1)
        order_within = np.argsort(-part_scores, axis=1)
        topk_idx = np.take_along_axis(part, order_within, axis=1)
        topk_scores = np.take_along_axis(part_scores, order_within, axis=1)
        lemmas_arr = np.asarray(model.bank.lemmas, dtype=object)
        topk_lemmas = lemmas_arr[topk_idx]
        for i in range(n):
            topk_log.append(list(zip(topk_lemmas[i].tolist(), topk_scores[i].tolist())))
        js = topk_idx[:, 0].tolist()
        ret_sims = topk_scores[:, 0].tolist()
        return js, ret_sims

    model._retrieve = patched_retrieve
    return topk_log


def build_items(units: list[tuple[str, list[str]]]):
    """units: list of (key, [words]). Returns (items, ranges) where ranges[i] = (start,end)
    indexes into items for units[i]."""
    items, ranges = [], []
    for key, words in units:
        start = len(items)
        sent = " ".join(words)
        pos = 0
        for w in words:
            idx = sent.find(w, pos)
            items.append({"form": w, "sentence": sent, "start": idx})
            pos = idx + len(w)
        ranges.append((start, len(items)))
    return items, ranges


@task(log_prints=True)
def load_shoshan_model(model_dir: str, bank_dir: str, device: str = "cpu"):
    from shoshan import Lemmatizer

    print(f"Loading Shoshan model from {model_dir}, bank from {bank_dir}")
    model = Lemmatizer(Path(model_dir), Path(bank_dir), device=device, blank_function_words=False)
    return model


def lemmatize_units(
    model,
    units: list[tuple[str, list[str]]],
    *,
    batch: int = 256,
    chunk_words: int = 50_000,
    on_progress=None,
    resume_all_topk: list | None = None,
    resume_done_items: int = 0,
    on_checkpoint=None,
) -> dict[str, list[list[tuple[str, float]]]]:
    """Returns {key: [per-word top-K [(lemma, score), ...]]}.

    resume_all_topk/resume_done_items let a caller resume a previously-interrupted
    run (e.g. after a spot-preempted pod) without redoing finished chunks.
    on_checkpoint(all_topk, done_items), if given, is called after every chunk so the
    caller can persist progress somewhere durable (pod-local disk does not survive
    eviction)."""
    topk_log = _patch_for_topk(model)
    items, ranges = build_items(units)
    all_topk: list = list(resume_all_topk) if resume_all_topk else []
    t0 = time.time()
    for lo in range(resume_done_items, len(items), chunk_words):
        hi = min(lo + chunk_words, len(items))
        del topk_log[:]
        model.lemmatize(items[lo:hi], batch=batch)
        all_topk.extend(topk_log)
        elapsed = time.time() - t0
        rate = (hi - resume_done_items) / max(elapsed, 0.01)
        print(f"  {hi}/{len(items)} words lemmatized ({elapsed:.0f}s elapsed this run, {rate:.0f} words/sec)")
        if on_progress:
            on_progress(hi, len(items))
        if on_checkpoint:
            on_checkpoint(all_topk, hi)
    out = {}
    for (key, words), (s, e) in zip(units, ranges):
        out[key] = all_topk[s:e]
    return out


def chunk_citing_corpus(corpus: dict[str, list[str]], chunk: int = 15) -> list[tuple[str, list[str]]]:
    units = []
    for title, words in corpus.items():
        for i in range(0, len(words), chunk):
            units.append((f"{title}#{i}", words[i : i + chunk]))
    return units
