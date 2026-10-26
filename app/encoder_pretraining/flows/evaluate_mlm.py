import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from prefect import task
from transformers import AutoTokenizer

from embeddings.steps.training.model_cache import cache_huggingface_model
from encoder_pretraining.steps.data import load_eval_windows
from encoder_pretraining.steps.data.dataset import MANIFEST_NAME, read_manifest
from encoder_pretraining.steps.evaluation import StaticMaskedDataset, evaluate_masked_lm, fill_mask_samples
from encoder_pretraining.steps.training import load_masked_lm, resolve_attention_implementation
from utils.gcs import download_blob_to_path, download_directory, write_blob_text
from utils.slack import notify_workflow_event, slack_notified_flow


WORKFLOW_NAME = "evaluate-mlm"


@task(log_prints=True)
def download_eval_corpus(bucket: str, prefix: str, local_dir: str) -> dict:
    """Download the manifest and only the eval shards (the train shards are large and not needed)."""
    prefix = prefix.rstrip("/")
    download_blob_to_path(bucket, f"{prefix}/{MANIFEST_NAME}", str(Path(local_dir) / MANIFEST_NAME))
    manifest = read_manifest(local_dir)
    for group, group_report in manifest["groups"].items():
        if not group.startswith("eval-"):
            continue
        for file_name in group_report["files"]:
            download_blob_to_path(bucket, f"{prefix}/{file_name}", str(Path(local_dir) / file_name))
    return manifest


@task(log_prints=True)
def cache_base_model(model_repo_id: str, target_dir: str, hub_cache_dir: str, revision: Optional[str]) -> str:
    cache_huggingface_model(
        model_repo_id=model_repo_id,
        target_dir=target_dir,
        hub_cache_dir=hub_cache_dir,
        revision=revision,
        force_download=False,
    )
    return target_dir


@task(log_prints=True)
def download_trained_model(bucket: str, prefix: str, local_dir: str) -> str:
    print(f"Downloading trained model from gs://{bucket}/{prefix}")
    return download_directory(bucket, prefix, local_dir)


@task(log_prints=True)
def evaluate_model(
    label: str,
    model_path: str,
    corpus_dir: str,
    manifest: dict,
    mlm_probability: float,
    eval_max_sequences: int,
    batch_size: int,
    seed: int,
    attn_implementation: str,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = load_masked_lm(
        model_path,
        resolve_attention_implementation(attn_implementation),
        reference_compile=False,
    )
    windows_by_group = load_eval_windows(corpus_dir, manifest, eval_max_sequences)

    per_source = {}
    for group, windows in windows_by_group.items():
        dataset = StaticMaskedDataset(windows, tokenizer, mlm_probability=mlm_probability, seed=seed)
        per_source[group.removeprefix("eval-")] = evaluate_masked_lm(model, dataset, batch_size)
        print(f"{label} {group}: {per_source[group.removeprefix('eval-')]}")

    overall_dataset = StaticMaskedDataset(
        np.concatenate(list(windows_by_group.values())), tokenizer, mlm_probability=mlm_probability, seed=seed
    )
    return {
        "overall": evaluate_masked_lm(model, overall_dataset, batch_size),
        "per_source": per_source,
        "fill_mask": fill_mask_samples(model, tokenizer),
    }


def compare_models(base: dict, trained: dict) -> dict:
    sources = ["overall", *base["per_source"]]
    comparison = {}
    for source in sources:
        base_metrics = base["overall"] if source == "overall" else base["per_source"][source]
        trained_metrics = trained["overall"] if source == "overall" else trained["per_source"][source]
        comparison[source] = {
            "base_loss": base_metrics["loss"],
            "trained_loss": trained_metrics["loss"],
            "loss_delta": trained_metrics["loss"] - base_metrics["loss"],
            "base_masked_accuracy": base_metrics["masked_accuracy"],
            "trained_masked_accuracy": trained_metrics["masked_accuracy"],
        }
    return comparison


@slack_notified_flow(workflow_name=WORKFLOW_NAME, log_prints=True)
def evaluate_mlm_flow(
    corpus_bucket: str,
    corpus_prefix: str,
    model_bucket: str,
    model_prefix: str,
    report_bucket: str,
    report_blob: str,
    base_model_repo_id: str = "answerdotai/ModernBERT-base",
    revision: Optional[str] = None,
    cache_root: str = "/cache/huggingface",
    mlm_probability: float = 0.3,
    eval_max_sequences: int = 5000,
    batch_size: int = 16,
    seed: int = 0,
    attn_implementation: str = "auto",
) -> None:
    safe_model_name = base_model_repo_id.replace("/", "__")
    base_model_dir = str(Path(cache_root) / "models" / safe_model_name)
    hub_cache_dir = str(Path(cache_root) / "hub")
    work_dir = tempfile.mkdtemp(dir="/tmp", prefix="evaluate-mlm-")
    corpus_dir = str(Path(work_dir) / "corpus")
    trained_dir = str(Path(work_dir) / "trained")

    try:
        manifest = download_eval_corpus(corpus_bucket, corpus_prefix, corpus_dir)
        cache_base_model(base_model_repo_id, base_model_dir, hub_cache_dir, revision)
        download_trained_model(model_bucket, model_prefix, trained_dir)

        shared = dict(
            corpus_dir=corpus_dir,
            manifest=manifest,
            mlm_probability=mlm_probability,
            eval_max_sequences=eval_max_sequences,
            batch_size=batch_size,
            seed=seed,
            attn_implementation=attn_implementation,
        )
        trained_result = evaluate_model(label="trained", model_path=trained_dir, **shared)
        # The eval windows are tokenized with the trained model's tokenizer. If its vocabulary was
        # extended, the base model cannot read them, and per-token loss would not be comparable anyway.
        extended_vocab = len(AutoTokenizer.from_pretrained(trained_dir)) != len(AutoTokenizer.from_pretrained(base_model_dir))
        if extended_vocab:
            print("Trained model has an extended vocabulary; skipping the base model comparison.")
            base_result = None
            comparison = None
        else:
            base_result = evaluate_model(label="base", model_path=base_model_dir, **shared)
            comparison = compare_models(base_result, trained_result)

        report = {
            "status": "success",
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "corpus_uri": f"gs://{corpus_bucket}/{corpus_prefix}",
            "trained_model_uri": f"gs://{model_bucket}/{model_prefix}",
            "base_model_repo_id": base_model_repo_id,
            "mlm_probability": mlm_probability,
            "seed": seed,
            "comparison": comparison,
            "base": base_result,
            "trained": trained_result,
        }
        write_blob_text(report_bucket, report_blob, json.dumps(report, ensure_ascii=False, indent=2))
        details = {"report_uri": f"gs://{report_bucket}/{report_blob}"}
        if comparison:
            overall = comparison["overall"]
            details.update(
                base_loss=round(overall["base_loss"], 4),
                trained_loss=round(overall["trained_loss"], 4),
                base_masked_accuracy=round(overall["base_masked_accuracy"], 4),
                trained_masked_accuracy=round(overall["trained_masked_accuracy"], 4),
            )
        else:
            overall = trained_result["overall"]
            details.update(trained_loss=round(overall["loss"], 4), trained_masked_accuracy=round(overall["masked_accuracy"], 4))
        notify_workflow_event(
            workflow_name=WORKFLOW_NAME,
            title=f"{WORKFLOW_NAME} completed",
            status="completed",
            details=details,
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
