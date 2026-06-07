import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from prefect import task

from embeddings.flows.evaluate_embeddings import (
    current_flow_run_id,
    download_eval_dataset_artifacts,
    download_huggingface_base_model,
    download_sentence_transformer_model,
    resolve_model_prefix,
    upload_evaluation_report,
)
from embeddings.steps.evaluation import evaluate_retrieval_dataset
from embeddings.steps.evaluation.llm_relevance import (
    LlmRelevanceAnalytics,
    LlmRelevanceConfig,
    evaluate_llm_judged_top_k,
)
from embeddings.steps.evaluation.pipeline import EvaluationConfig
from embeddings.steps.query_generation.cache import flush_cache
from utils.slack import SlackProgressReporter, notify_workflow_event, slack_notified_flow


def llm_relevance_progress_details(snapshot: dict) -> dict:
    estimated_cost = snapshot.get("estimated_cost", {})
    cache = snapshot.get("cache", {})
    llm = snapshot.get("llm", {})
    return {
        "Relevant judgments": snapshot.get("relevant_judgments", 0),
        "Cache hits": cache.get("hits", 0),
        "Cache misses": cache.get("misses", 0),
        "Remote requests": llm.get("remote_requests_succeeded", 0),
        "Failures": llm.get("remote_failures", 0),
        "Estimated remote cost": f"${estimated_cost.get('remote_estimated_cost_usd', 0.0):.6f}",
    }


@task(log_prints=True)
def run_retrieval_ranking(
    documents_path: str,
    queries_path: str,
    qrels_path: str,
    metadata_path: str,
    output_dir: str,
    evaluation_backend: str,
    sentence_transformer_model_path: Optional[str],
    sentence_transformer_batch_size: int,
    normalize_sentence_transformer_embeddings: bool,
    gemini_cache_path: str,
    gemini_cache_enabled: bool,
    gemini_max_workers: int,
    top_k: int,
) -> dict:
    normalized_evaluation_backend = evaluation_backend.lower()
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if normalized_evaluation_backend == "gemini" and not api_key:
        raise ValueError("Missing GOOGLE_API_KEY or GEMINI_API_KEY for Gemini retrieval ranking.")

    report = evaluate_retrieval_dataset(
        documents_path=documents_path,
        queries_path=queries_path,
        qrels_path=qrels_path,
        metadata_path=metadata_path,
        output_dir=output_dir,
        config=EvaluationConfig(
            model_source=normalized_evaluation_backend,
            sentence_transformer_model_path=sentence_transformer_model_path,
            sentence_transformer_batch_size=sentence_transformer_batch_size,
            normalize_sentence_transformer_embeddings=normalize_sentence_transformer_embeddings,
            gemini_api_key=api_key,
            gemini_cache_enabled=gemini_cache_enabled,
            gemini_cache_path=gemini_cache_path,
            gemini_max_workers=gemini_max_workers,
            top_k_results=top_k,
        ),
    )
    summary = report["summary"]
    dataset = report.get("dataset") or {}
    role_counts = dataset.get("retrieval_role_counts") or {}
    print(
        "Retrieval ranking completed for LLM relevance evaluation: "
        f"backend={summary['backend']}, "
        f"queries={summary['query_count']}, "
        f"documents={dataset.get('documents_count', 0)}, "
        f"positive_documents={role_counts.get('positive', 0)}, "
        f"distractor_documents={role_counts.get('distractor', 0)}, "
        f"top_k={top_k}"
    )
    return report


@task(log_prints=True)
def run_llm_relevance_judging(
    documents_path: str,
    retrieval_results_path: str,
    retrieval_report_path: str,
    output_dir: str,
    top_k: int,
    judge_model: str,
    judge_max_workers: int,
    judge_cache_path: str,
    judge_cache_enabled: bool,
    flush_judge_cache: bool,
    workflow_name: str,
) -> dict:
    if flush_judge_cache:
        print(f"Flushing LLM relevance cache at {judge_cache_path}")
        flush_cache(judge_cache_path)

    retrieval_rows = sum(1 for line in Path(retrieval_results_path).open("r") if line.strip())
    total_jobs = retrieval_rows * top_k
    analytics = LlmRelevanceAnalytics()
    reporter = SlackProgressReporter(
        workflow_name=workflow_name,
        total_units=total_jobs,
        unit_label="judgments",
        notify_every_fraction=0.1,
    )
    reporter.notify_start(
        {
            "Queries": retrieval_rows,
            "Top K": top_k,
            "Judge model": judge_model,
            "Max workers": judge_max_workers,
            "Cache path": judge_cache_path,
            "Flush cache": flush_judge_cache,
        }
    )

    def progress_callback(completed_jobs: int, _total_jobs: int, snapshot: dict) -> None:
        reporter.notify_progress_if_due(completed_jobs, llm_relevance_progress_details(snapshot))
        if completed_jobs % max(1, total_jobs // 20) == 0 or completed_jobs == total_jobs:
            print(
                "LLM relevance progress: "
                f"judgments={completed_jobs}/{total_jobs}, "
                f"relevant={snapshot.get('relevant_judgments', 0)}, "
                f"cache_hits={snapshot.get('cache', {}).get('hits', 0)}, "
                f"cache_misses={snapshot.get('cache', {}).get('misses', 0)}, "
                f"remote_requests={snapshot.get('llm', {}).get('remote_requests_succeeded', 0)}, "
                f"failures={snapshot.get('llm', {}).get('remote_failures', 0)}, "
                f"estimated_remote_cost_usd={snapshot.get('estimated_cost', {}).get('remote_estimated_cost_usd', 0.0):.6f}"
            )

    report = evaluate_llm_judged_top_k(
        documents_path=documents_path,
        retrieval_results_path=retrieval_results_path,
        retrieval_report_path=retrieval_report_path,
        output_dir=output_dir,
        config=LlmRelevanceConfig(
            model=judge_model,
            top_k=top_k,
            max_workers=judge_max_workers,
            cache_enabled=judge_cache_enabled,
            cache_path=judge_cache_path,
            runtime_analytics=analytics,
            progress_callback=progress_callback,
        ),
    )
    summary = report["summary"]
    reporter.notify_success(
        {
            "Queries": summary["query_count"],
            "Judged pairs": summary["judged_pairs"],
            f"Relevant count@{top_k}": f"{summary[f'llm_relevant_count@{top_k}']:.6f}",
            f"Precision@{top_k}": f"{summary[f'llm_precision@{top_k}']:.6f}",
            f"Any relevant@{top_k}": f"{summary[f'llm_any_relevant_rate@{top_k}']:.6f}",
            "Failures": report.get("judge", {}).get("failures_count", 0),
        }
    )
    return report


@slack_notified_flow(workflow_name="evaluate-embeddings-llm-relevance", log_prints=True)
def evaluate_embeddings_llm_relevance_flow(
    eval_dataset_bucket: str,
    eval_dataset_prefix: str,
    report_bucket: str,
    report_prefix: str,
    evaluation_backend: Literal["latest", "gemini", "custom", "base"] = "latest",
    model_bucket: str = "development-research",
    latest_model_prefix: str = "custom_embeddings/models/berel_sentence_transformer/latest",
    gcs_model_prefix: Optional[str] = None,
    sentence_transformer_batch_size: int = 32,
    normalize_sentence_transformer_embeddings: bool = True,
    gemini_cache_path: str = "/cache/evaluation/gemini_embedding_cache.sqlite",
    gemini_cache_enabled: bool = True,
    gemini_max_workers: int = 4,
    base_model_repo_id: str = "dicta-il/BEREL_3.0",
    base_model_hub_cache_dir: str = "/cache/huggingface",
    top_k: int = 10,
    judge_model: str = "claude-sonnet-4-6",
    judge_max_workers: int = 4,
    judge_cache_path: str = "/cache/evaluation/llm_relevance_cache.sqlite",
    judge_cache_enabled: bool = True,
    flush_judge_cache: bool = False,
    workflow_name: str = "evaluate-embeddings-llm-relevance",
) -> None:
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    run_id = current_flow_run_id()
    output_dir = tempfile.mkdtemp(dir="/tmp", prefix="embedding-llm-relevance-")
    model_dir = tempfile.mkdtemp(dir="/tmp", prefix="embedding-llm-relevance-model-")
    run_report_prefix = f"{report_prefix.rstrip('/')}/runs/{run_id}"
    latest_report_prefix = f"{report_prefix.rstrip('/')}/latest"
    downloaded_artifacts = {}
    try:
        downloaded_artifacts = download_eval_dataset_artifacts(eval_dataset_bucket, eval_dataset_prefix)
        model_prefix = resolve_model_prefix(evaluation_backend, latest_model_prefix, gcs_model_prefix)
        sentence_transformer_model_path = None
        if model_prefix is not None:
            sentence_transformer_model_path = download_sentence_transformer_model(model_bucket, model_prefix, model_dir)
        elif evaluation_backend.lower() == "base":
            sentence_transformer_model_path = download_huggingface_base_model(base_model_repo_id, base_model_hub_cache_dir)

        started_at = datetime.now(timezone.utc).isoformat()
        retrieval_report = run_retrieval_ranking(
            documents_path=downloaded_artifacts["documents.jsonl"],
            queries_path=downloaded_artifacts["queries.jsonl"],
            qrels_path=downloaded_artifacts["qrels.jsonl"],
            metadata_path=downloaded_artifacts["metadata.json"],
            output_dir=output_dir,
            evaluation_backend=evaluation_backend,
            sentence_transformer_model_path=sentence_transformer_model_path,
            sentence_transformer_batch_size=sentence_transformer_batch_size,
            normalize_sentence_transformer_embeddings=normalize_sentence_transformer_embeddings,
            gemini_cache_path=gemini_cache_path,
            gemini_cache_enabled=gemini_cache_enabled,
            gemini_max_workers=gemini_max_workers,
            top_k=top_k,
        )
        dataset = retrieval_report.get("dataset") or {}
        role_counts = dataset.get("retrieval_role_counts") or {}
        notify_workflow_event(
            workflow_name=workflow_name,
            title=f"{workflow_name} ranking completed",
            status="ranking completed",
            details={
                "evaluation_backend": evaluation_backend,
                "queries": retrieval_report["summary"]["query_count"],
                "corpus_documents": dataset.get("documents_count", 0),
                "positive_documents": role_counts.get("positive", 0),
                "distractor_documents": role_counts.get("distractor", 0),
                "top_k": top_k,
            },
        )

        report = run_llm_relevance_judging(
            documents_path=downloaded_artifacts["documents.jsonl"],
            retrieval_results_path=str(Path(output_dir) / "per_query_results.jsonl"),
            retrieval_report_path=str(Path(output_dir) / "run_config.json"),
            output_dir=output_dir,
            top_k=top_k,
            judge_model=judge_model,
            judge_max_workers=judge_max_workers,
            judge_cache_path=judge_cache_path,
            judge_cache_enabled=judge_cache_enabled,
            flush_judge_cache=flush_judge_cache,
            workflow_name=workflow_name,
        )
        report["workflow"] = {
            "run_id": run_id,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "eval_dataset_bucket": eval_dataset_bucket,
            "eval_dataset_prefix": eval_dataset_prefix,
            "evaluation_backend": evaluation_backend,
            "model_bucket": model_bucket,
            "model_prefix": model_prefix,
            "report_bucket": report_bucket,
            "report_run_prefix": run_report_prefix,
            "report_latest_prefix": latest_report_prefix,
        }
        Path(output_dir, "run_config.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        run_upload = upload_evaluation_report(output_dir, report_bucket, run_report_prefix)
        latest_upload = upload_evaluation_report(output_dir, report_bucket, latest_report_prefix)
        print(f"Uploaded LLM relevance reports to {run_upload['uri']} and {latest_upload['uri']}")
        notify_workflow_event(
            workflow_name=workflow_name,
            title=f"{workflow_name} reports uploaded",
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
