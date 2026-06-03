import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from prefect import task
from prefect.context import MissingContextError, get_run_context

from embeddings.steps.evaluation import evaluate_retrieval_dataset
from embeddings.steps.evaluation.pipeline import EvaluationConfig
from utils.gcs import download_blob, download_directory, upload_directory
from utils.slack import notify_workflow_event, slack_notified_flow


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


@task(log_prints=True)
def download_eval_dataset_artifacts(bucket: str, prefix: str) -> dict:
    artifacts = {}
    for name in ("documents.jsonl", "queries.jsonl", "qrels.jsonl", "metadata.json"):
        blob_path = f"{prefix.rstrip('/')}/{name}"
        print(f"Downloading gs://{bucket}/{blob_path}")
        artifacts[name] = download_blob(bucket, blob_path, local_dir="/tmp")
    return artifacts


@task(log_prints=True)
def download_sentence_transformer_model(bucket: str, prefix: str, local_dir: str) -> str:
    print(f"Downloading SentenceTransformer model from gs://{bucket}/{prefix}")
    return download_directory(bucket, prefix, local_dir)


@task(log_prints=True)
def run_embedding_evaluation(
    documents_path: str,
    queries_path: str,
    qrels_path: str,
    metadata_path: str,
    output_dir: str,
    model_source: str,
    sentence_transformer_model_path: Optional[str],
    sentence_transformer_batch_size: int,
    normalize_sentence_transformer_embeddings: bool,
    gemini_cache_path: str,
    gemini_cache_enabled: bool,
    gemini_max_workers: int,
) -> dict:
    normalized_model_source = model_source.lower()
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if normalized_model_source == "gemini" and not api_key:
        raise ValueError("Missing GOOGLE_API_KEY or GEMINI_API_KEY for Gemini evaluation.")

    config = EvaluationConfig(
        model_source=normalized_model_source,
        sentence_transformer_model_path=sentence_transformer_model_path,
        sentence_transformer_batch_size=sentence_transformer_batch_size,
        normalize_sentence_transformer_embeddings=normalize_sentence_transformer_embeddings,
        gemini_api_key=api_key,
        gemini_cache_enabled=gemini_cache_enabled,
        gemini_cache_path=gemini_cache_path,
        gemini_max_workers=gemini_max_workers,
    )
    report = evaluate_retrieval_dataset(
        documents_path=documents_path,
        queries_path=queries_path,
        qrels_path=qrels_path,
        metadata_path=metadata_path,
        output_dir=output_dir,
        config=config,
    )
    summary = report["summary"]
    print(
        "Embedding evaluation summary: "
        f"model_source={model_source}, "
        f"backend={summary['backend']}, "
        f"queries={summary['query_count']}, "
        f"ndcg@10={summary['ndcg@10']:.6f}, "
        f"recall@10={summary['recall@10']:.6f}, "
        f"recall@50={summary['recall@50']:.6f}, "
        f"mrr@10={summary['mrr@10']:.6f}"
    )
    notify_workflow_event(
        workflow_name="evaluate-embeddings",
        title="evaluate-embeddings completed",
        status="completed",
        details={
            "model_source": model_source,
            "backend": summary["backend"],
            "queries": summary["query_count"],
            "ndcg@10": f"{summary['ndcg@10']:.6f}",
            "recall@10": f"{summary['recall@10']:.6f}",
            "recall@50": f"{summary['recall@50']:.6f}",
            "mrr@10": f"{summary['mrr@10']:.6f}",
        },
    )
    return report


@task(log_prints=True)
def upload_evaluation_report(local_dir: str, bucket: str, prefix: str) -> dict:
    print(f"Uploading evaluation report directory to gs://{bucket}/{prefix}")
    upload_directory(local_dir, bucket, prefix)
    return {
        "bucket": bucket,
        "prefix": prefix,
        "uri": f"gs://{bucket}/{prefix}",
    }


def resolve_model_prefix(model_source: str, latest_model_prefix: str, gcs_model_prefix: Optional[str]) -> Optional[str]:
    normalized_model_source = model_source.lower()
    if normalized_model_source == "latest":
        return latest_model_prefix
    if normalized_model_source == "gcs":
        if not gcs_model_prefix:
            raise ValueError("gcs_model_prefix is required when model_source='gcs'.")
        return gcs_model_prefix
    if normalized_model_source == "gemini":
        return None
    raise ValueError("model_source must be one of: latest, gcs, gemini.")


@slack_notified_flow(workflow_name="evaluate-embeddings", log_prints=True)
def evaluate_embeddings_flow(
    eval_dataset_bucket: str,
    eval_dataset_prefix: str,
    report_bucket: str,
    report_prefix: str,
    model_source: str = "latest",
    model_bucket: str = "development-research",
    latest_model_prefix: str = "custom_embeddings/models/berel_sentence_transformer/latest",
    gcs_model_prefix: Optional[str] = None,
    sentence_transformer_batch_size: int = 32,
    normalize_sentence_transformer_embeddings: bool = True,
    gemini_cache_path: str = "/cache/evaluation/gemini_embedding_cache.sqlite",
    gemini_cache_enabled: bool = True,
    gemini_max_workers: int = 4,
) -> None:
    run_id = current_flow_run_id()
    output_dir = tempfile.mkdtemp(dir="/tmp", prefix="embedding-eval-")
    model_dir = tempfile.mkdtemp(dir="/tmp", prefix="embedding-eval-model-")
    run_report_prefix = f"{report_prefix.rstrip('/')}/runs/{run_id}"
    latest_report_prefix = f"{report_prefix.rstrip('/')}/latest"
    downloaded_artifacts = {}
    try:
        downloaded_artifacts = download_eval_dataset_artifacts(eval_dataset_bucket, eval_dataset_prefix)
        model_prefix = resolve_model_prefix(model_source, latest_model_prefix, gcs_model_prefix)
        sentence_transformer_model_path = None
        if model_prefix is not None:
            sentence_transformer_model_path = download_sentence_transformer_model(model_bucket, model_prefix, model_dir)

        started_at = datetime.now(timezone.utc).isoformat()
        report = run_embedding_evaluation(
            documents_path=downloaded_artifacts["documents.jsonl"],
            queries_path=downloaded_artifacts["queries.jsonl"],
            qrels_path=downloaded_artifacts["qrels.jsonl"],
            metadata_path=downloaded_artifacts["metadata.json"],
            output_dir=output_dir,
            model_source=model_source,
            sentence_transformer_model_path=sentence_transformer_model_path,
            sentence_transformer_batch_size=sentence_transformer_batch_size,
            normalize_sentence_transformer_embeddings=normalize_sentence_transformer_embeddings,
            gemini_cache_path=gemini_cache_path,
            gemini_cache_enabled=gemini_cache_enabled,
            gemini_max_workers=gemini_max_workers,
        )
        report["workflow"] = {
            "run_id": run_id,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "eval_dataset_bucket": eval_dataset_bucket,
            "eval_dataset_prefix": eval_dataset_prefix,
            "model_source": model_source,
            "model_bucket": model_bucket,
            "model_prefix": model_prefix,
            "report_bucket": report_bucket,
            "report_run_prefix": run_report_prefix,
            "report_latest_prefix": latest_report_prefix,
        }
        Path(output_dir, "run_config.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        run_upload = upload_evaluation_report(output_dir, report_bucket, run_report_prefix)
        latest_upload = upload_evaluation_report(output_dir, report_bucket, latest_report_prefix)
        print(f"Uploaded evaluation reports to {run_upload['uri']} and {latest_upload['uri']}")
        notify_workflow_event(
            workflow_name="evaluate-embeddings",
            title="evaluate-embeddings reports uploaded",
            status="uploaded",
            details={
                "run_report": run_upload["uri"],
                "latest_report": latest_upload["uri"],
            },
        )
    finally:
        for local_path in downloaded_artifacts.values():
            Path(local_path).unlink(missing_ok=True)
        shutil.rmtree(output_dir, ignore_errors=True)
        shutil.rmtree(model_dir, ignore_errors=True)
