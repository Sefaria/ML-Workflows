import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import hf_hub_download
from prefect import task

from embeddings.steps.training.model_cache import cache_huggingface_model
from roshei_tevot.steps.data import DATASET_REPO, load_split
from roshei_tevot.steps.evaluation import evaluate_split
from roshei_tevot.steps.index import ExpansionIndex
from roshei_tevot.steps.model import load_for_inference
from utils.gcs import download_directory, write_blob_text
from utils.slack import notify_workflow_event, slack_notified_flow


WORKFLOW_NAME = "evaluate-roshei-tevot"
LEXICON_FILE = "lexicon/surface_index.jsonl"
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
def evaluate_model(label: str, model_path: str, lexicon_path: str, dataset_repo: str, splits: list[str], max_rows: int | None, scan_unknown: bool) -> dict:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, config = load_for_inference(model_path, device)
    index = ExpansionIndex.build(lexicon_path, model, tokenizer, device)
    settings = {k: config[k] for k in ("max_length", "context_words") if k in config}
    report = {}
    for split in splits:
        rows = load_split(split, dataset_repo, settings.get("context_words", 64), max_rows)
        print(f"{label}: {split} ({len(rows)} rows)")
        report[split] = evaluate_split(model, tokenizer, rows, index, device, scan_unknown, **settings)
    return report


@slack_notified_flow(workflow_name=WORKFLOW_NAME, log_prints=True)
def evaluate_roshei_tevot_flow(
    models: list[dict],
    report_bucket: str,
    report_blob: str,
    dataset_repo: str = DATASET_REPO,
    lexicon_file: str = LEXICON_FILE,
    cache_root: str = "/cache/huggingface",
    splits: list[str] | None = None,
    max_rows: int | None = None,
    scan_unknown: bool = False,
) -> None:
    """Rank candidates for each held-out split with every model. List the untrained base model first to get the no-fine-tuning baseline."""
    work_dir = tempfile.mkdtemp(dir="/tmp", prefix="roshei-tevot-eval-")
    try:
        lexicon_path = hf_hub_download(dataset_repo, lexicon_file, repo_type="dataset")
        results = {}
        for spec in models:
            path = fetch_model(spec, cache_root, work_dir)
            results[spec["label"]] = evaluate_model(spec["label"], path, lexicon_path, dataset_repo, splits or SPLITS, max_rows, scan_unknown)
        report = {"status": "success", "evaluated_at": datetime.now(timezone.utc).isoformat(), "dataset": dataset_repo, "models": results, "specs": models}
        write_blob_text(report_bucket, report_blob, json.dumps(report, ensure_ascii=False, indent=2))
        details = {"report_uri": f"gs://{report_bucket}/{report_blob}"}
        for label, by_split in results.items():
            for split, r in by_split.items():
                o = r["index"]["overall"]
                details[f"{label}/{split}"] = f"top1={o.get('top1', 0):.3f} mrr={o.get('mrr', 0):.3f}"
        notify_workflow_event(workflow_name=WORKFLOW_NAME, title=f"{WORKFLOW_NAME} completed", status="completed", details=details)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
