import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from prefect import task

from embeddings.steps.training.model_cache import cache_huggingface_model
from roshei_tevot.steps.data import DATASET_REPO, attach_candidates, load_hebrew_names, load_lexicon, load_split
from roshei_tevot.steps.evaluation import PRIOR_WEIGHTS, evaluate_splits
from roshei_tevot.steps.model import load_for_inference
from utils.gcs import download_directory, write_blob_text
from utils.slack import notify_workflow_event, slack_notified_flow


WORKFLOW_NAME = "evaluate-roshei-tevot"
SPLITS = ["val", "test_seen_keys", "test_unseen_keys"]


@task(log_prints=True)
def fetch_model(spec: dict, cache_root: str, work_dir: str) -> str:
    """A model spec is {"label", "hf_repo"} or {"label", "bucket", "prefix"}."""
    if spec.get("hf_repo"):
        target = str(Path(cache_root) / "models" / spec["hf_repo"].replace("/", "__"))
        cache_huggingface_model(spec["hf_repo"], target, str(Path(cache_root) / "hub"), spec.get("revision"), False)
        return target
    return download_directory(spec["bucket"], spec["prefix"], str(Path(work_dir) / spec["label"]))


@task(log_prints=True)
def prepare_rows(dataset_repo: str, splits: list[str], max_rows: Optional[int], min_candidates: int, context_words: int) -> tuple[dict, dict]:
    lexicon, names = load_lexicon(dataset_repo), load_hebrew_names(dataset_repo)
    rows, reports = {}, {}
    for split in splits:
        rows[split], reports[split] = attach_candidates(load_split(split, dataset_repo, context_words, None, names), lexicon, min_candidates, training=False)
        rows[split] = rows[split][:max_rows] if max_rows else rows[split]
        print(f"{split}: {reports[split]}")
    return rows, reports


@task(log_prints=True)
def evaluate_model(label: str, model_path: str, split_rows: dict, prior_weights: list[float]) -> dict:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, config = load_for_inference(model_path, device)
    settings = {k: config[k] for k in ("max_length", "context_words") if k in config}
    report = evaluate_splits(model, tokenizer, split_rows, device, float(config.get("logit_scale", 1.0)), tuple(prior_weights), **settings)
    report["architecture"] = config["architecture"]
    print(f"{label}: λ={report['prior_weight']} success={report['success']['passes']}")
    return report


def gaps(results: dict, baseline: str) -> dict:
    """Top-1 of each model minus the baseline model's, per split, at each model's chosen λ."""
    base = results[baseline]["splits"]
    return {
        label: {split: r["splits"][split]["overall"].get("top1", 0.0) - base[split]["overall"].get("top1", 0.0) for split in r["splits"] if split in base}
        for label, r in results.items() if label != baseline
    }


@slack_notified_flow(workflow_name=WORKFLOW_NAME, log_prints=True)
def evaluate_roshei_tevot_flow(
    models: list[dict],
    report_bucket: str,
    report_blob: str,
    dataset_repo: str = DATASET_REPO,
    cache_root: str = "/cache/huggingface",
    splits: Optional[list[str]] = None,
    max_rows: Optional[int] = None,
    min_candidates: int = 2,
    context_words: int = 100,
    prior_weights: Optional[list[float]] = None,
) -> None:
    """Rank each row's candidates with every model. `gap_to_first` compares every model with the first in the list, so list the
    siamese model first to read the cross-encoder's advantage directly."""
    work_dir = tempfile.mkdtemp(dir="/tmp", prefix="roshei-tevot-eval-")
    try:
        split_rows, candidate_reports = prepare_rows(dataset_repo, splits or SPLITS, max_rows, min_candidates, context_words)
        results = {}
        for spec in models:
            results[spec["label"]] = evaluate_model(spec["label"], fetch_model(spec, cache_root, work_dir), split_rows, prior_weights or list(PRIOR_WEIGHTS))
        report = {"status": "success", "evaluated_at": datetime.now(timezone.utc).isoformat(), "dataset": dataset_repo, "specs": models,
                  "candidates": candidate_reports, "models": results, "gap_to_first": gaps(results, models[0]["label"])}
        write_blob_text(report_bucket, report_blob, json.dumps(report, ensure_ascii=False, indent=2))
        details = {"report_uri": f"gs://{report_bucket}/{report_blob}"}
        for label, r in results.items():
            for split, s in r["splits"].items():
                o = s["overall"]
                details[f"{label}/{split}"] = f"top1={o.get('top1', 0):.3f} freq={o.get('frequency_top1', 0):.3f} λ={r['prior_weight']}"
            details[f"{label}/success"] = r["success"]["passes"]
        notify_workflow_event(workflow_name=WORKFLOW_NAME, title=f"{WORKFLOW_NAME} completed", status="completed", details=details)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
