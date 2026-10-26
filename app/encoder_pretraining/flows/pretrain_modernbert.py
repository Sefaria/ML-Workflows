import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from prefect import task

from embeddings.steps.training.model_cache import cache_huggingface_model
from encoder_pretraining.steps.training import pretrain_masked_lm, require_bf16_gpu, validate_modernbert
from utils.gcs import blob_exists, download_directory, upload_directory, write_blob_text
from utils.slack import notify_workflow_event, slack_notified_flow


WORKFLOW_NAME = "pretrain-modernbert"
SUCCESS_MARKER = "_SUCCESS"


@task(log_prints=True)
def check_accelerator() -> dict:
    report = require_bf16_gpu()
    print(f"Training accelerator: {report}")
    notify_workflow_event(
        workflow_name=WORKFLOW_NAME,
        title=f"{WORKFLOW_NAME} accelerator detected",
        status="runtime",
        details={
            "gpu": report["name"],
            "compute_capability": report["compute_capability"],
            "device_count": report["device_count"],
            "cuda_version": report["cuda_version"],
        },
    )
    return report


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
    return validate_modernbert(model_path)


@task(log_prints=True)
def download_corpus(bucket: str, prefix: str, local_dir: str) -> str:
    print(f"Downloading packed corpus from gs://{bucket}/{prefix}")
    return download_directory(bucket, prefix, local_dir)


@task(log_prints=True)
def train_model(**kwargs) -> dict:
    return pretrain_masked_lm(**kwargs)


@task(log_prints=True)
def upload_final_model(model_dir: str, bucket: str, prefix: str) -> dict:
    print(f"Uploading final model to gs://{bucket}/{prefix}")
    upload_directory(model_dir, bucket, prefix)
    # Written last so a half-uploaded model is never mistaken for a finished run.
    write_blob_text(bucket, f"{prefix}/{SUCCESS_MARKER}", json.dumps({"finished_at": datetime.now(timezone.utc).isoformat()}))
    return {"bucket": bucket, "prefix": prefix, "uri": f"gs://{bucket}/{prefix}"}


@task(log_prints=True)
def upload_training_report(report: dict, bucket: str, blob_path: str) -> str:
    print(f"Uploading training report to gs://{bucket}/{blob_path}")
    write_blob_text(bucket, blob_path, json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return f"gs://{bucket}/{blob_path}"


@slack_notified_flow(workflow_name=WORKFLOW_NAME, log_prints=True)
def pretrain_modernbert_flow(
    run_name: str,
    corpus_bucket: str,
    corpus_prefix: str,
    model_bucket: str,
    model_prefix: str,
    model_repo_id: str = "answerdotai/ModernBERT-base",
    revision: Optional[str] = None,
    cache_root: str = "/cache/huggingface",
    force_download: bool = False,
    mlm_probability: float = 0.3,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.05,
    lr_scheduler_type: str = "linear",
    epochs: float = 3.0,
    max_tokens: Optional[int] = None,
    global_batch_sequences: int = 512,
    per_device_batch_size: int = 32,
    eval_steps: int = 250,
    eval_max_sequences: int = 2000,
    checkpoint_minutes: float = 10.0,
    attn_implementation: str = "auto",
    reference_compile: bool = False,
    seed: int = 0,
    dataloader_num_workers: int = 0,
) -> None:
    # run_name is stable across restarts, so a restarted job finds its own checkpoints.
    run_root = f"{model_prefix.rstrip('/')}/{run_name}"
    final_prefix = f"{run_root}/final"
    if blob_exists(model_bucket, f"{final_prefix}/{SUCCESS_MARKER}"):
        print(f"Run {run_name} already finished at gs://{model_bucket}/{final_prefix}; nothing to do.")
        return

    safe_model_name = model_repo_id.replace("/", "__")
    base_model_dir = str(Path(cache_root) / "models" / safe_model_name)
    hub_cache_dir = str(Path(cache_root) / "hub")
    work_dir = tempfile.mkdtemp(dir="/tmp", prefix="modernbert-pretrain-")
    corpus_dir = str(Path(work_dir) / "corpus")
    output_dir = str(Path(work_dir) / "output")
    started_at = datetime.now(timezone.utc).isoformat()

    try:
        accelerator_report = check_accelerator()
        cache_report = cache_base_model(model_repo_id, base_model_dir, hub_cache_dir, revision, force_download)
        validation_report = validate_base_model(base_model_dir)
        download_corpus(corpus_bucket, corpus_prefix, corpus_dir)
        training_report = train_model(
            corpus_dir=corpus_dir,
            base_model_path=base_model_dir,
            output_dir=output_dir,
            checkpoint_bucket=model_bucket,
            checkpoint_prefix=f"{run_root}/checkpoints",
            mlm_probability=mlm_probability,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler_type,
            epochs=epochs,
            max_tokens=max_tokens,
            global_batch_sequences=global_batch_sequences,
            per_device_batch_size=per_device_batch_size,
            eval_steps=eval_steps,
            eval_max_sequences=eval_max_sequences,
            checkpoint_minutes=checkpoint_minutes,
            attn_implementation=attn_implementation,
            reference_compile=reference_compile,
            seed=seed,
            dataloader_num_workers=dataloader_num_workers,
            workflow_name=WORKFLOW_NAME,
        )
        model_upload = upload_final_model(training_report["final_model_dir"], model_bucket, final_prefix)
        report = {
            "status": "success",
            "run_name": run_name,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "corpus_uri": f"gs://{corpus_bucket}/{corpus_prefix}",
            "model_repo_id": model_repo_id,
            "revision": revision,
            "accelerator": accelerator_report,
            "cache": cache_report,
            "validation": validation_report,
            "training": training_report,
            "model_upload": model_upload,
            "hyperparameters": {
                "mlm_probability": mlm_probability,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "warmup_ratio": warmup_ratio,
                "lr_scheduler_type": lr_scheduler_type,
                "epochs": epochs,
                "max_tokens": max_tokens,
                "global_batch_sequences": global_batch_sequences,
                "per_device_batch_size": per_device_batch_size,
                "seed": seed,
            },
        }
        report_uri = upload_training_report(report, model_bucket, f"{run_root}/training_report.json")
        notify_workflow_event(
            workflow_name=WORKFLOW_NAME,
            title=f"{WORKFLOW_NAME} completed",
            status="completed",
            details={
                "run_name": run_name,
                "model_uri": model_upload["uri"],
                "report_uri": report_uri,
                "eval_loss": training_report["final_eval"].get("eval_loss"),
                "mean_tokens_per_second": training_report["mean_tokens_per_second"],
                "attn_implementation": training_report["attn_implementation"],
            },
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
