import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from prefect import task
from transformers import AutoTokenizer

from embeddings.steps.training.model_cache import cache_huggingface_model
from encoder_pretraining.steps.data import hebrew_tokenization_report, prepare_corpus
from encoder_pretraining.steps.data.dataset import MANIFEST_NAME
from encoder_pretraining.steps.training import validate_modernbert
from utils.gcs import download_blob, upload_directory
from utils.slack import notify_workflow_event, slack_notified_flow


PROGRESS_LOG_EVERY_DOCUMENTS = 20000


@task(log_prints=True)
def cache_base_model(model_repo_id: str, target_dir: str, hub_cache_dir: str, revision: Optional[str]) -> dict:
    print(f"Ensuring base model {model_repo_id} is cached at {target_dir}")
    return cache_huggingface_model(
        model_repo_id=model_repo_id,
        target_dir=target_dir,
        hub_cache_dir=hub_cache_dir,
        revision=revision,
        force_download=False,
    )


@task(log_prints=True)
def validate_base_model(model_path: str) -> dict:
    print(f"Validating base model can be loaded from {model_path}")
    return validate_modernbert(model_path)


@task(log_prints=True)
def download_training_data(bucket: str, blob_path: str) -> str:
    print(f"Downloading training data from gs://{bucket}/{blob_path}")
    return download_blob(bucket, blob_path, local_dir="/tmp")


@task(log_prints=True)
def measure_hebrew_tokenization(documents_path: str, model_path: str) -> dict:
    report = hebrew_tokenization_report(documents_path, AutoTokenizer.from_pretrained(model_path))
    print(f"Hebrew tokenization report: {report}")
    return report


@task(log_prints=True)
def tokenize_and_pack(
    documents_path: str,
    model_path: str,
    output_dir: str,
    seq_len: int,
    shard_sequences: int,
    document_limit: Optional[int],
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    next_log = PROGRESS_LOG_EVERY_DOCUMENTS

    def log_progress(documents_seen: int) -> None:
        nonlocal next_log
        if documents_seen >= next_log:
            print(f"Tokenized {documents_seen} documents")
            next_log += PROGRESS_LOG_EVERY_DOCUMENTS

    return prepare_corpus(
        documents_path=documents_path,
        tokenizer=tokenizer,
        output_dir=output_dir,
        seq_len=seq_len,
        shard_sequences=shard_sequences,
        document_limit=document_limit,
        progress_callback=log_progress,
    )


@task(log_prints=True)
def upload_corpus(output_dir: str, bucket: str, prefix: str) -> dict:
    print(f"Uploading packed corpus to gs://{bucket}/{prefix}")
    upload_directory(output_dir, bucket, prefix)
    return {"bucket": bucket, "prefix": prefix, "uri": f"gs://{bucket}/{prefix}"}


def corpus_summary(packing: dict) -> dict:
    train = packing["groups"].get("train", {})
    return {
        "train_sequences": train.get("sequences"),
        "train_tokens": train.get("tokens"),
        "eval_groups": ", ".join(name for name in packing["groups"] if name.startswith("eval-")),
        "tokenized_train_tokens": packing["stats"].get("train", {}).get("tokens"),
        "stored_train_tokens": packing["stats"].get("train", {}).get("stored_tokens"),
    }


@slack_notified_flow(workflow_name="prepare-mlm-corpus", log_prints=True)
def prepare_mlm_corpus_flow(
    source_bucket: str,
    source_blob: str,
    dest_bucket: str,
    dest_prefix: str,
    model_repo_id: str = "answerdotai/ModernBERT-base",
    revision: Optional[str] = None,
    cache_root: str = "/cache/huggingface",
    max_seq_length: int = 1024,
    shard_sequences: int = 100000,
    document_limit: Optional[int] = None,
) -> None:
    safe_model_name = model_repo_id.replace("/", "__")
    base_model_dir = str(Path(cache_root) / "models" / safe_model_name)
    hub_cache_dir = str(Path(cache_root) / "hub")
    output_dir = tempfile.mkdtemp(dir="/tmp", prefix="mlm-corpus-")
    started_at = datetime.now(timezone.utc).isoformat()

    documents_path = None
    try:
        cache_base_model(model_repo_id, base_model_dir, hub_cache_dir, revision)
        validation_report = validate_base_model(base_model_dir)
        documents_path = download_training_data(source_bucket, source_blob)
        hebrew_report = measure_hebrew_tokenization(documents_path, base_model_dir)
        packing = tokenize_and_pack(
            documents_path=documents_path,
            model_path=base_model_dir,
            output_dir=output_dir,
            seq_len=max_seq_length,
            shard_sequences=shard_sequences,
            document_limit=document_limit,
        )
        manifest = {
            "status": "success",
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "source_bucket": source_bucket,
            "source_blob": source_blob,
            "model_repo_id": model_repo_id,
            "revision": revision,
            "document_limit": document_limit,
            "validation": validation_report,
            "hebrew_tokenization": hebrew_report,
            **packing,
        }
        (Path(output_dir) / MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        upload = upload_corpus(output_dir, dest_bucket, dest_prefix)
        notify_workflow_event(
            workflow_name="prepare-mlm-corpus",
            title="prepare-mlm-corpus completed",
            status="completed",
            details={
                "corpus_uri": upload["uri"],
                "mean_tokens_per_hebrew_word": hebrew_report.get("mean_tokens_per_hebrew_word"),
                **corpus_summary(packing),
            },
        )
    finally:
        if documents_path:
            Path(documents_path).unlink(missing_ok=True)
        shutil.rmtree(output_dir, ignore_errors=True)
