import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from prefect import task
from prefect.context import MissingContextError, get_run_context

from embeddings.steps.evaluation import BayesianComparisonConfig, compare_evaluation_reports
from utils.gcs import download_blob, upload_directory
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
def download_evaluation_report_artifacts(bucket: str, prefix: str, label: str) -> dict:
    artifacts = {}
    for name in ("per_query_results.jsonl", "results_summary.json", "run_config.json"):
        blob_path = f"{prefix.rstrip('/')}/{name}"
        print(f"Downloading {label} evaluation artifact gs://{bucket}/{blob_path}")
        artifacts[name] = download_blob(bucket, blob_path, local_dir="/tmp")
    return artifacts


@task(log_prints=True)
def run_bayesian_evaluation_comparison(
    a_artifacts: dict,
    b_artifacts: dict,
    output_dir: str,
    metric: str,
    rope: float,
    posterior_draws: int,
    seed: int,
    credible_interval_mass: float,
    confidence_level: float,
    a_label: Optional[str],
    b_label: Optional[str],
) -> dict:
    config = BayesianComparisonConfig(
        metric=metric,
        rope=rope,
        posterior_draws=posterior_draws,
        seed=seed,
        credible_interval_mass=credible_interval_mass,
        confidence_level=confidence_level,
    )
    report = compare_evaluation_reports(
        a_per_query_path=a_artifacts["per_query_results.jsonl"],
        b_per_query_path=b_artifacts["per_query_results.jsonl"],
        a_summary_path=a_artifacts["results_summary.json"],
        b_summary_path=b_artifacts["results_summary.json"],
        a_run_config_path=a_artifacts["run_config.json"],
        b_run_config_path=b_artifacts["run_config.json"],
        output_dir=output_dir,
        config=config,
        a_label=a_label,
        b_label=b_label,
    )
    posterior = report["posterior"]
    ci = posterior["credible_interval"]
    print(
        "Bayesian embedding comparison summary: "
        f"metric={metric}, "
        f"paired_queries={report['paired_query_count']}, "
        f"mean_diff={posterior['posterior_mean_difference']:.6f}, "
        f"ci=[{ci['lower']:.6f}, {ci['upper']:.6f}], "
        f"p_a_practically_better={posterior['p_a_practically_better']:.4f}, "
        f"p_equivalent={posterior['p_practically_equivalent']:.4f}, "
        f"p_b_practically_better={posterior['p_b_practically_better']:.4f}, "
        f"conclusion={posterior['conclusion']}"
    )
    notify_workflow_event(
        workflow_name="compare-embedding-evaluations",
        title="compare-embedding-evaluations completed",
        status="completed",
        details={
            "metric": metric,
            "paired_queries": report["paired_query_count"],
            "a": report["a_label"],
            "b": report["b_label"],
            "mean_diff": f"{posterior['posterior_mean_difference']:.6f}",
            "credible_interval": f"[{ci['lower']:.6f}, {ci['upper']:.6f}]",
            "P(A > rope)": f"{posterior['p_a_practically_better']:.4f}",
            "P(eq)": f"{posterior['p_practically_equivalent']:.4f}",
            "P(B > rope)": f"{posterior['p_b_practically_better']:.4f}",
            "conclusion": posterior["conclusion"],
        },
    )
    return report


@task(log_prints=True)
def upload_comparison_report(local_dir: str, bucket: str, prefix: str) -> dict:
    print(f"Uploading comparison report directory to gs://{bucket}/{prefix}")
    upload_directory(local_dir, bucket, prefix)
    return {
        "bucket": bucket,
        "prefix": prefix,
        "uri": f"gs://{bucket}/{prefix}",
    }


@slack_notified_flow(workflow_name="compare-embedding-evaluations", log_prints=True)
def compare_embedding_evaluations_flow(
    evaluation_a_bucket: str = "development-research",
    evaluation_a_prefix: Optional[str] = None,
    evaluation_b_bucket: str = "development-research",
    evaluation_b_prefix: Optional[str] = None,
    report_bucket: str = "development-research",
    report_prefix: str = "custom_embeddings/eval/embedding_evaluation_comparisons",
    metric: str = "ndcg@10",
    rope: float = 0.005,
    posterior_draws: int = 20000,
    seed: int = 613,
    credible_interval_mass: float = 0.95,
    confidence_level: float = 0.95,
    evaluation_a_label: Optional[str] = None,
    evaluation_b_label: Optional[str] = None,
) -> None:
    if not evaluation_a_prefix:
        raise ValueError("evaluation_a_prefix is required.")
    if not evaluation_b_prefix:
        raise ValueError("evaluation_b_prefix is required.")

    run_id = current_flow_run_id()
    output_dir = tempfile.mkdtemp(dir="/tmp", prefix="embedding-eval-comparison-")
    run_report_prefix = f"{report_prefix.rstrip('/')}/runs/{run_id}"
    latest_report_prefix = f"{report_prefix.rstrip('/')}/latest"
    a_artifacts = {}
    b_artifacts = {}
    try:
        a_artifacts = download_evaluation_report_artifacts(evaluation_a_bucket, evaluation_a_prefix, "A")
        b_artifacts = download_evaluation_report_artifacts(evaluation_b_bucket, evaluation_b_prefix, "B")
        report = run_bayesian_evaluation_comparison(
            a_artifacts=a_artifacts,
            b_artifacts=b_artifacts,
            output_dir=output_dir,
            metric=metric,
            rope=rope,
            posterior_draws=posterior_draws,
            seed=seed,
            credible_interval_mass=credible_interval_mass,
            confidence_level=confidence_level,
            a_label=evaluation_a_label,
            b_label=evaluation_b_label,
        )
        report["workflow"] = {
            "run_id": run_id,
            "evaluation_a_bucket": evaluation_a_bucket,
            "evaluation_a_prefix": evaluation_a_prefix,
            "evaluation_b_bucket": evaluation_b_bucket,
            "evaluation_b_prefix": evaluation_b_prefix,
            "report_bucket": report_bucket,
            "report_run_prefix": run_report_prefix,
            "report_latest_prefix": latest_report_prefix,
        }
        Path(output_dir, "comparison_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        run_upload = upload_comparison_report(output_dir, report_bucket, run_report_prefix)
        latest_upload = upload_comparison_report(output_dir, report_bucket, latest_report_prefix)
        print(f"Uploaded comparison reports to {run_upload['uri']} and {latest_upload['uri']}")
        notify_workflow_event(
            workflow_name="compare-embedding-evaluations",
            title="compare-embedding-evaluations reports uploaded",
            status="uploaded",
            details={
                "run_report": run_upload["uri"],
                "latest_report": latest_upload["uri"],
            },
        )
    finally:
        for local_path in list(a_artifacts.values()) + list(b_artifacts.values()):
            Path(local_path).unlink(missing_ok=True)
        shutil.rmtree(output_dir, ignore_errors=True)
