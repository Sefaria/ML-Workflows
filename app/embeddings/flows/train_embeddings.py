import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from prefect import task
from prefect.context import MissingContextError, get_run_context

from embeddings.steps.training import (
    cache_huggingface_model,
    train_from_query_dataset_artifacts,
    validate_masked_lm_model,
)
from utils.gcs import delete_prefix, download_blob, upload_blob, upload_directory
from utils.slack import notify_training_validation_metric, slack_notified_flow


@task(log_prints=True)
def cache_base_model(
    model_repo_id: str,
    target_dir: str,
    hub_cache_dir: str,
    revision: Optional[str],
    force_download: bool,
) -> dict:
    print(f"Ensuring base model {model_repo_id} is cached at {target_dir}")
    return cache_huggingface_model(
        model_repo_id=model_repo_id,
        target_dir=target_dir,
        hub_cache_dir=hub_cache_dir,
        revision=revision,
        force_download=force_download,
    )


@task(log_prints=True)
def validate_base_model(model_path: str) -> dict:
    print(f"Validating base model can be loaded from {model_path}")
    return validate_masked_lm_model(model_path)


@task(log_prints=True)
def download_query_dataset_artifacts(bucket: str, prefix: str) -> dict:
    artifacts = {}
    for name in ("documents.jsonl", "queries.jsonl", "qrels.jsonl"):
        blob_path = f"{prefix}/{name}"
        print(f"Downloading gs://{bucket}/{blob_path}")
        artifacts[name] = download_blob(bucket, blob_path, local_dir="/tmp")
    return artifacts


@task(log_prints=True)
def train_model(
    documents_path: str,
    queries_path: str,
    qrels_path: str,
    base_model_path: str,
    output_dir: str,
    max_seq_length: int,
    normalize_embeddings: bool,
    batch_size: int,
    epochs: int,
    warmup_steps: Optional[int],
    learning_rate: float,
    max_examples: Optional[int],
    use_amp: Optional[bool],
    validation_fraction: float,
    validation_seed: int,
    evaluation_steps: Optional[int],
) -> dict:
    print(f"Training SentenceTransformer model from base model {base_model_path}")

    def report_validation_metric(metrics: dict[str, float], epoch: float, steps: int) -> None:
        metric_values = {
            "ndcg@10": metrics.get("validation_cosine_ndcg@10"),
            "mrr@10": metrics.get("validation_cosine_mrr@10"),
            "recall@10": metrics.get("validation_cosine_recall@10"),
            "recall@50": metrics.get("validation_cosine_recall@50"),
        }
        compact_metric_values = {
            name: f"{value:.6f}"
            for name, value in metric_values.items()
            if value is not None
        }
        print(
            f"Training validation metric: epoch={epoch:.4g}/{epochs}, "
            f"steps={steps}, "
            + ", ".join(f"{name}={value}" for name, value in compact_metric_values.items())
        )
        notify_training_validation_metric(
            workflow_name="train-embeddings",
            epoch=epoch,
            total_epochs=epochs,
            steps=steps,
            metrics=compact_metric_values,
            details={
                "selection_metric": "ndcg@10",
                "base_model": base_model_path,
            },
        )

    return train_from_query_dataset_artifacts(
        documents_path=documents_path,
        queries_path=queries_path,
        qrels_path=qrels_path,
        base_model_path=base_model_path,
        output_dir=output_dir,
        max_seq_length=max_seq_length,
        normalize_embeddings=normalize_embeddings,
        batch_size=batch_size,
        epochs=epochs,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        max_examples=max_examples,
        use_amp=use_amp,
        validation_fraction=validation_fraction,
        validation_seed=validation_seed,
        evaluation_steps=evaluation_steps,
        evaluation_callback=report_validation_metric,
    )


@task(log_prints=True)
def upload_trained_model(model_dir: str, bucket: str, run_prefix: str, latest_prefix: str) -> dict:
    print(f"Uploading trained model to run path gs://{bucket}/{run_prefix}")
    upload_directory(model_dir, bucket, run_prefix)

    print(f"Replacing latest trained model at gs://{bucket}/{latest_prefix}")
    deleted_latest_files = delete_prefix(bucket, latest_prefix)
    upload_directory(model_dir, bucket, latest_prefix)
    return {
        "bucket": bucket,
        "run_prefix": run_prefix,
        "latest_prefix": latest_prefix,
        "run_uri": f"gs://{bucket}/{run_prefix}",
        "latest_uri": f"gs://{bucket}/{latest_prefix}",
        "deleted_latest_files": deleted_latest_files,
    }


@task(log_prints=True)
def upload_training_report(report: dict, bucket: str, run_blob_path: str, latest_blob_path: str) -> dict:
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
        json.dump(report, tmp, ensure_ascii=False, indent=2)
        local_path = tmp.name

    try:
        print(f"Uploading training report to run path gs://{bucket}/{run_blob_path}")
        upload_blob(local_path, bucket, run_blob_path)
        print(f"Uploading latest training report to gs://{bucket}/{latest_blob_path}")
        upload_blob(local_path, bucket, latest_blob_path)
        return {
            "bucket": bucket,
            "run_blob": run_blob_path,
            "latest_blob": latest_blob_path,
            "run_uri": f"gs://{bucket}/{run_blob_path}",
            "latest_uri": f"gs://{bucket}/{latest_blob_path}",
        }
    finally:
        Path(local_path).unlink(missing_ok=True)


def current_flow_run_id() -> str:
    try:
        context = get_run_context()
    except MissingContextError:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    flow_run = getattr(context, "flow_run", None)
    flow_run_id = getattr(flow_run, "id", None)
    if flow_run_id:
        return str(flow_run_id)
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def training_output_paths(model_prefix: str, report_blob: str, run_id: str) -> dict:
    model_root = model_prefix.rstrip("/")
    report_path = Path(report_blob)
    latest_report_blob = str(report_path)
    run_report_blob = str(report_path.with_name(f"{report_path.stem}-{run_id}{report_path.suffix}"))
    return {
        "run_id": run_id,
        "model_run_prefix": f"{model_root}/runs/{run_id}",
        "model_latest_prefix": f"{model_root}/latest",
        "report_run_blob": run_report_blob,
        "report_latest_blob": latest_report_blob,
    }


@slack_notified_flow(workflow_name="train-embeddings", log_prints=True)
def train_embeddings_flow(
    source_bucket: str,
    source_prefix: str,
    model_bucket: str,
    model_prefix: str,
    report_bucket: str,
    report_blob: str,
    model_repo_id: str = "dicta-il/BEREL_3.0",
    revision: Optional[str] = None,
    cache_root: str = "/cache/huggingface",
    force_download: bool = False,
    validate_model: bool = True,
    max_seq_length: int = 512,
    normalize_embeddings: bool = True,
    batch_size: int = 16,
    epochs: int = 3,
    warmup_steps: Optional[int] = None,
    learning_rate: float = 2e-5,
    max_examples: Optional[int] = None,
    use_amp: Optional[bool] = None,
    validation_fraction: float = 0.1,
    validation_seed: int = 0,
    evaluation_steps: Optional[int] = None,
) -> None:
    safe_model_name = model_repo_id.replace("/", "__")
    base_model_dir = str(Path(cache_root) / "models" / safe_model_name)
    hub_cache_dir = str(Path(cache_root) / "hub")
    output_dir = tempfile.mkdtemp(dir="/tmp", prefix="trained-embeddings-")
    run_id = current_flow_run_id()
    output_paths = training_output_paths(model_prefix, report_blob, run_id)

    started_at = datetime.now(timezone.utc).isoformat()
    downloaded_artifacts = {}
    try:
        cache_report = cache_base_model(
            model_repo_id=model_repo_id,
            target_dir=base_model_dir,
            hub_cache_dir=hub_cache_dir,
            revision=revision,
            force_download=force_download,
        )
        validation_report = validate_base_model(base_model_dir) if validate_model else None
        downloaded_artifacts = download_query_dataset_artifacts(source_bucket, source_prefix)
        training_report = train_model(
            documents_path=downloaded_artifacts["documents.jsonl"],
            queries_path=downloaded_artifacts["queries.jsonl"],
            qrels_path=downloaded_artifacts["qrels.jsonl"],
            base_model_path=base_model_dir,
            output_dir=output_dir,
            max_seq_length=max_seq_length,
            normalize_embeddings=normalize_embeddings,
            batch_size=batch_size,
            epochs=epochs,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            max_examples=max_examples,
            use_amp=use_amp,
            validation_fraction=validation_fraction,
            validation_seed=validation_seed,
            evaluation_steps=evaluation_steps,
        )
        model_upload_report = upload_trained_model(
            output_dir,
            model_bucket,
            output_paths["model_run_prefix"],
            output_paths["model_latest_prefix"],
        )
        expected_report_upload = {
            "bucket": report_bucket,
            "run_blob": output_paths["report_run_blob"],
            "latest_blob": output_paths["report_latest_blob"],
            "run_uri": f"gs://{report_bucket}/{output_paths['report_run_blob']}",
            "latest_uri": f"gs://{report_bucket}/{output_paths['report_latest_blob']}",
        }
        finished_at = datetime.now(timezone.utc).isoformat()
        report = {
            "status": "success",
            "workflow_stage": "embedding_training",
            "run_id": run_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "source_bucket": source_bucket,
            "source_prefix": source_prefix,
            "model_bucket": model_bucket,
            "model_prefix": model_prefix,
            "model_run_prefix": output_paths["model_run_prefix"],
            "model_latest_prefix": output_paths["model_latest_prefix"],
            "model_upload": model_upload_report,
            "model_repo_id": model_repo_id,
            "revision": revision,
            "report_bucket": report_bucket,
            "report_blob": report_blob,
            "report_run_blob": output_paths["report_run_blob"],
            "report_latest_blob": output_paths["report_latest_blob"],
            "cache_root": cache_root,
            "base_model_dir": base_model_dir,
            "hub_cache_dir": hub_cache_dir,
            "force_download": force_download,
            "validate_model": validate_model,
            "cache": cache_report,
            "validation": validation_report,
            "training": training_report,
            "report_upload": expected_report_upload,
            "environment": {
                "HF_HOME": os.getenv("HF_HOME"),
                "HF_HUB_CACHE": os.getenv("HF_HUB_CACHE"),
                "TRANSFORMERS_CACHE": os.getenv("TRANSFORMERS_CACHE"),
            },
        }
        upload_training_report(
            report,
            report_bucket,
            output_paths["report_run_blob"],
            output_paths["report_latest_blob"],
        )
    finally:
        for local_path in downloaded_artifacts.values():
            Path(local_path).unlink(missing_ok=True)
        shutil.rmtree(output_dir, ignore_errors=True)
