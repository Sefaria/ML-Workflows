import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from huggingface_hub import hf_hub_download
from prefect import task

from embeddings.steps.training.model_cache import cache_huggingface_model
from encoder_pretraining.steps.evaluation import cross_validated_probe
from utils.gcs import download_directory, write_blob_text
from utils.slack import notify_workflow_event, slack_notified_flow


WORKFLOW_NAME = "evaluate-ner-probe"


@task(log_prints=True)
def load_gold_examples(dataset_repo: str, filename: str) -> list[dict]:
    path = hf_hub_download(dataset_repo, filename, repo_type="dataset")
    examples = json.loads(Path(path).read_text())
    print(f"Loaded {len(examples)} examples from {dataset_repo}")
    return [{"text": e["text"], "spans": e["spans"]} for e in examples]


@task(log_prints=True)
def fetch_model(spec: dict, cache_root: str, work_dir: str) -> str:
    """A model spec is {"label", "hf_repo"} or {"label", "bucket", "prefix"}."""
    if spec.get("hf_repo"):
        target = str(Path(cache_root) / "models" / spec["hf_repo"].replace("/", "__"))
        cache_huggingface_model(spec["hf_repo"], target, str(Path(cache_root) / "hub"), spec.get("revision"), False)
        return target
    return download_directory(spec["bucket"], spec["prefix"], str(Path(work_dir) / spec["label"]))


@task(log_prints=True)
def probe_model(label: str, model_path: str, examples: list[dict], **settings) -> list[dict]:
    print(f"NER probe for {label}")
    return cross_validated_probe(model_path, examples, **settings)


def summarize(runs: list[dict]) -> dict:
    def collect(name: str, metric: str) -> np.ndarray:
        return np.array([run["scores"][name][metric] for run in runs])

    return {
        name: {"f1_mean": float(collect(name, "f1").mean()), "f1_std": float(collect(name, "f1").std(ddof=1)) if len(runs) > 1 else 0.0,
               "precision_mean": float(collect(name, "precision").mean()), "recall_mean": float(collect(name, "recall").mean())}
        for name in ("micro", "Person", "Citation")
    }


def paired_difference(candidate: list[dict], baseline: list[dict]) -> dict:
    """Mean and standard error of the F1 gain over the baseline, paired by (seed, fold)."""
    out = {}
    for name in ("micro", "Person", "Citation"):
        diffs = np.array([c["scores"][name]["f1"] - b["scores"][name]["f1"] for c, b in zip(candidate, baseline)])
        out[name] = {"mean": float(diffs.mean()), "stderr": float(diffs.std(ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else 0.0, "wins": int((diffs > 0).sum()), "runs": len(diffs)}
    return out


@slack_notified_flow(workflow_name=WORKFLOW_NAME, log_prints=True)
def evaluate_ner_probe_flow(
    models: list[dict],
    report_bucket: str,
    report_blob: str,
    dataset_repo: str = "Sefaria/en_linker_gold_dataset",
    dataset_filename: str = "en_linker_gold_dataset.json",
    cache_root: str = "/cache/huggingface",
    folds: int = 5,
    seeds: Optional[list[int]] = None,
    epochs: int = 8,
    learning_rate: float = 5e-5,
    batch_size: int = 16,
    max_length: int = 512,
    attn_implementation: str = "auto",
) -> None:
    """Fine-tune each model on the same cross-validation splits of the gold Person/Citation set.

    The first model in the list is the baseline: every other model is compared with it per fold.
    """
    seeds = seeds or [0, 1]
    work_dir = tempfile.mkdtemp(dir="/tmp", prefix="ner-probe-")
    try:
        examples = load_gold_examples(dataset_repo, dataset_filename)
        results = {}
        for spec in models:
            path = fetch_model(spec, cache_root, work_dir)
            results[spec["label"]] = probe_model(
                spec["label"], path, examples,
                folds=folds, seeds=seeds, epochs=epochs, learning_rate=learning_rate,
                batch_size=batch_size, max_length=max_length, attn_implementation=attn_implementation,
            )
        baseline_label = models[0]["label"]
        report = {
            "status": "success",
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "dataset": f"{dataset_repo}/{dataset_filename}",
            "examples": len(examples),
            "settings": {"folds": folds, "seeds": seeds, "epochs": epochs, "learning_rate": learning_rate, "batch_size": batch_size, "max_length": max_length},
            "baseline": baseline_label,
            "models": {
                label: {"spec": next(m for m in models if m["label"] == label), "summary": summarize(runs),
                        "vs_baseline": None if label == baseline_label else paired_difference(runs, results[baseline_label]), "runs": runs}
                for label, runs in results.items()
            },
        }
        write_blob_text(report_bucket, report_blob, json.dumps(report, ensure_ascii=False, indent=2))
        details = {"report_uri": f"gs://{report_bucket}/{report_blob}"}
        for label in results:
            summary = report["models"][label]["summary"]["micro"]
            details[f"{label}_f1"] = f"{summary['f1_mean']:.4f} +/- {summary['f1_std']:.4f}"
        notify_workflow_event(workflow_name=WORKFLOW_NAME, title=f"{WORKFLOW_NAME} completed", status="completed", details=details)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
