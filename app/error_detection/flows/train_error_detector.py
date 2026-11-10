import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from prefect import task

from embeddings.steps.training.model_cache import cache_huggingface_model
from encoder_pretraining.steps.training import require_bf16_gpu
from error_detection.steps.data import GCS_DATASET
from error_detection.steps.training import run_training
from utils.gcs import blob_exists, upload_directory, write_blob_text
from utils.slack import notify_workflow_event, slack_notified_flow


WORKFLOW_NAME = "train-error-detector"
SUCCESS_MARKER = "_SUCCESS"


@task(log_prints=True)
def check_accelerator() -> dict:
    report = require_bf16_gpu()
    print(f"Training accelerator: {report}")
    return report


@task(log_prints=True)
def cache_base_model(model_repo_id: str, target_dir: str, hub_cache_dir: str, revision: Optional[str]) -> dict:
    return cache_huggingface_model(model_repo_id=model_repo_id, target_dir=target_dir, hub_cache_dir=hub_cache_dir, revision=revision, force_download=False)


@task(log_prints=True)
def train(**kwargs) -> dict:
    return run_training(**kwargs)


@task(log_prints=True)
def upload_final_model(model_dir: str, bucket: str, prefix: str) -> str:
    upload_directory(model_dir, bucket, prefix)
    # Written last so a half-uploaded model is never mistaken for a finished run.
    write_blob_text(bucket, f"{prefix}/{SUCCESS_MARKER}", json.dumps({"finished_at": datetime.now(timezone.utc).isoformat()}))
    return f"gs://{bucket}/{prefix}"


@slack_notified_flow(workflow_name=WORKFLOW_NAME, log_prints=True)
def train_error_detector_flow(
    run_name: str,
    model_bucket: str,
    model_prefix: str,
    model_repo_id: str = "dicta-il/BEREL_3.0",
    revision: Optional[str] = None,
    dataset_repo: str = GCS_DATASET,
    cache_root: str = "/cache/huggingface",
    max_length: int = 256,
    paired_negatives: bool = True,
    negative_weight: float = 1.0,
    synthetic_weight: float = 0.5,
    pos_weight: float = 5.0,
    samples_per_epoch: Optional[int] = None,
    epochs: float = 3.0,
    batch_size: int = 64,
    learning_rate: float = 3e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.05,
    eval_steps: int = 500,
    checkpoint_minutes: float = 10.0,
    seed: int = 0,
    dataloader_num_workers: int = 4,
    max_train_rows: Optional[int] = None,
    label_scope: str = "all",
) -> None:
    """Fine-tune BEREL as a binary token classifier (error / ok) on Sefaria/hebrew-text-corrections. Keeps the weights with the best
    validation PR-AUC. run_name is stable across restarts, so a restarted job finds its own checkpoints.
    label_scope "segment" keeps neighbouring segments as context only (no labels) in real windows; see steps.data.chunk_row."""
    run_root = f"{model_prefix.rstrip('/')}/{run_name}"
    final_prefix = f"{run_root}/final"
    if blob_exists(model_bucket, f"{final_prefix}/{SUCCESS_MARKER}"):
        print(f"Run {run_name} already finished at gs://{model_bucket}/{final_prefix}; nothing to do.")
        return

    base_dir = str(Path(cache_root) / "models" / model_repo_id.replace("/", "__"))
    work_dir = tempfile.mkdtemp(dir="/tmp", prefix="error-detection-")
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        accelerator = check_accelerator()
        cache_base_model(model_repo_id, base_dir, str(Path(cache_root) / "hub"), revision)
        training = train(
            base_model_path=base_dir, output_dir=str(Path(work_dir) / "output"),
            checkpoint_bucket=model_bucket, checkpoint_prefix=f"{run_root}/checkpoints", workflow_name=WORKFLOW_NAME,
            dataset_repo=dataset_repo, max_length=max_length, paired_negatives=paired_negatives, negative_weight=negative_weight,
            synthetic_weight=synthetic_weight, pos_weight=pos_weight, samples_per_epoch=samples_per_epoch, epochs=epochs,
            batch_size=batch_size, learning_rate=learning_rate, weight_decay=weight_decay, warmup_ratio=warmup_ratio,
            eval_steps=eval_steps, checkpoint_minutes=checkpoint_minutes, seed=seed, dataloader_num_workers=dataloader_num_workers,
            max_train_rows=max_train_rows, label_scope=label_scope,
        )
        model_uri = upload_final_model(training["final_model_dir"], model_bucket, final_prefix)
        report = {"status": "success", "run_name": run_name, "started_at": started_at, "finished_at": datetime.now(timezone.utc).isoformat(),
                  "model_repo_id": model_repo_id, "revision": revision, "accelerator": accelerator, "training": training, "model_uri": model_uri}
        write_blob_text(model_bucket, f"{run_root}/training_report.json", json.dumps(report, ensure_ascii=False, indent=2, default=str))
        best = training["best_eval"]
        notify_workflow_event(workflow_name=WORKFLOW_NAME, title=f"{WORKFLOW_NAME} completed", status="completed",
                              details={"run_name": run_name, "model_uri": model_uri, "best_step": training["best_step"],
                                       "pr_auc": best.get("best_pr_auc", best.get("eval_pr_auc")),
                                       "f1": best.get("best_f1", best.get("eval_f1"))})
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
