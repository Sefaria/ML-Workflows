import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from prefect import task

from embeddings.steps.training.model_cache import cache_huggingface_model
from error_detection.steps.data import GCS_DATASET, build_training_rows, chunk_rows, load_rows
from error_detection.steps.evaluation import (build_lexicon, false_positive_sample, lexicon_scores, mlm_word_scores, predict_words,
                                              summarise)
from error_detection.steps.model import load_for_inference
from utils.gcs import download_directory, write_blob_text
from utils.slack import notify_workflow_event, slack_notified_flow


WORKFLOW_NAME = "evaluate-error-detector"
BASELINE_MODEL = "dicta-il/BEREL_3.0"


@task(log_prints=True)
def fetch_model(spec: dict, cache_root: str, work_dir: str) -> str:
    """A model spec is {"label", "bucket", "prefix"} (a trained run's final directory)."""
    return download_directory(spec["bucket"], spec["prefix"], str(Path(work_dir) / spec["label"]))


@task(log_prints=True)
def load_splits(dataset_repo: str, max_rows: Optional[int]) -> dict:
    # Each real window is evaluated with its corrected copy (all words 0), as in training validation, so the threshold also sees clean text.
    rows = {split: build_training_rows(load_rows("real", split, dataset_repo, max_rows), [], paired_negatives=True) for split in ("validation", "test")}
    rows["train_corrected"] = [row for row in build_training_rows(load_rows("real", "train", dataset_repo), [], paired_negatives=True)
                               if row["source"] == "corrected"]
    print({k: len(v) for k, v in rows.items()})
    return rows


def first_windows(rows: list[dict], n: Optional[int]) -> list[dict]:
    """The first `n` real windows and their corrected copies."""
    if not n:
        return rows
    keep = set(list(row["id"] for row in rows if row["source"] == "real")[:n])
    return [row for row in rows if row["id"].removesuffix(":corrected") in keep]


def _score_and_summarise(words_val: list[dict], words_test: list[dict], val_scores=None, test_scores=None) -> dict:
    """Threshold chosen on validation by best F1, then applied to test."""
    val = summarise(words_val, scores=val_scores)
    test = summarise(words_test, threshold=val["threshold"], scores=test_scores)
    return {"validation": val, "test": test}


@task(log_prints=True)
def evaluate_model(label: str, model_path: str, rows: dict, max_length: int, fp_sample: int) -> dict:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, config = load_for_inference(model_path, device)
    chunks = {split: chunk_rows(tokenizer, rows[split], config.get("max_length", max_length)) for split in ("validation", "test")}
    words = {split: predict_words(model, tokenizer, chunks[split], device) for split in chunks}
    report = _score_and_summarise(words["validation"], words["test"])
    rows_by_id = {row["id"]: row for row in rows["test"]}
    report["false_positive_sample"] = false_positive_sample(words["test"], rows_by_id, report["test"]["threshold"], fp_sample)
    report["config"] = config
    print(f"{label}: test pr_auc={report['test']['pr_auc']:.3f} f1={report['test']['f1']:.3f} "
          f"p={report['test']['precision']:.3f} r={report['test']['recall']:.3f}")
    return report


@task(log_prints=True)
def evaluate_baselines(rows: dict, cache_root: str, max_length: int, mlm_max_rows: Optional[int]) -> dict:
    """Lexicon (letter skeleton unseen in corrected training text) and BEREL masked-LM (lowest piece log-prob of the word)."""
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = str(Path(cache_root) / "models" / BASELINE_MODEL.replace("/", "__"))
    cache_huggingface_model(BASELINE_MODEL, base, str(Path(cache_root) / "hub"), None, False)
    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True)
    mlm = AutoModelForMaskedLM.from_pretrained(base).to(device).eval()

    out = {}
    lexicon = build_lexicon(rows["train_corrected"])
    chunks = {s: chunk_rows(tokenizer, first_windows(rows[s], mlm_max_rows), max_length) for s in ("validation", "test")}
    gold = {s: predict_words_labels(chunks[s]) for s in chunks}
    by_id = {s: {row["id"]: row for row in rows[s]} for s in ("validation", "test")}
    out["lexicon"] = _score_and_summarise(gold["validation"], gold["test"],
                                          lexicon_scores(chunks["validation"], by_id["validation"], lexicon),
                                          lexicon_scores(chunks["test"], by_id["test"], lexicon))
    out["berel_mlm"] = _score_and_summarise(gold["validation"], gold["test"],
                                            mlm_word_scores(mlm, tokenizer, chunks["validation"], device),
                                            mlm_word_scores(mlm, tokenizer, chunks["test"], device))
    out["lexicon"]["size"] = len(lexicon)
    for name, r in out.items():
        print(f"baseline {name}: test pr_auc={r['test']['pr_auc']:.3f} f1={r['test']['f1']:.3f}")
    return out


def predict_words_labels(chunks: list[dict]) -> list[dict]:
    """Word records without model scores, in the same order as predict_words (for baselines)."""
    out = []
    for chunk in chunks:
        positions = [i for i, label in enumerate(chunk["labels"]) if label != -100]
        for w, p in enumerate(positions):
            out.append({"chunk": chunk["id"], "word_index": chunk["word_index"][w], "label": int(chunk["labels"][p]),
                        "kind": chunk["word_kind"][w], "version": chunk["version"], "score": 0.0})
    return out


@slack_notified_flow(workflow_name=WORKFLOW_NAME, log_prints=True)
def evaluate_error_detector_flow(
    models: list[dict],
    report_bucket: str,
    report_blob: str,
    dataset_repo: str = GCS_DATASET,
    cache_root: str = "/cache/huggingface",
    max_rows: Optional[int] = None,
    max_length: int = 256,
    baselines: bool = True,
    mlm_max_rows: Optional[int] = 600,
    fp_sample: int = 100,
) -> None:
    """Word-level error-class metrics on the real validation and test splits (threshold chosen on validation), recall by fix kind, the
    lexicon and BEREL masked-LM baselines, and a sample of flagged words labelled 0 for hand audit. Synthetic data is never used here.
    The masked-LM baseline costs one forward pass per word piece, so it runs on the first `mlm_max_rows` rows of each split."""
    work_dir = tempfile.mkdtemp(dir="/tmp", prefix="error-detection-eval-")
    try:
        rows = load_splits(dataset_repo, max_rows)
        results = {}
        for spec in models:
            results[spec["label"]] = evaluate_model(spec["label"], fetch_model(spec, cache_root, work_dir), rows, max_length, fp_sample)
        report = {"models": results, "dataset_repo": dataset_repo}
        if baselines:
            report["baselines"] = evaluate_baselines(rows, cache_root, max_length, mlm_max_rows)
        write_blob_text(report_bucket, report_blob, json.dumps(report, ensure_ascii=False, indent=2, default=str))
        notify_workflow_event(workflow_name=WORKFLOW_NAME, title=f"{WORKFLOW_NAME} completed", status="completed",
                              details={"report": f"gs://{report_bucket}/{report_blob}",
                                       **{f"{k}_test_f1": round(v["test"]["f1"], 3) for k, v in results.items()}})
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
