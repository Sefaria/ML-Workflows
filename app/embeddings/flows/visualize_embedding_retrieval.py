import json
import tempfile
from pathlib import Path
from urllib.parse import quote

from prefect import task

from embeddings.steps.evaluation.retrieval_report import write_retrieval_visualization_pdf
from utils.gcs import download_blob, upload_blob
from utils.slack import SlackWebhookClient, prefect_flow_run_url, slack_notified_flow, workflow_icon_url


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _gcs_uri(bucket: str, blob_path: str) -> str:
    return f"gs://{bucket}/{blob_path}"


def _gcs_console_url(bucket: str, blob_path: str) -> str:
    return f"https://console.cloud.google.com/storage/browser/_details/{quote(bucket)}/{quote(blob_path, safe='/')}"


def _escape_mrkdwn(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _notify_pdf_available(bucket: str, blob_path: str, summary: dict, workflow_name: str) -> None:
    gs_uri = _gcs_uri(bucket, blob_path)
    console_url = _gcs_console_url(bucket, blob_path)
    run_url = prefect_flow_run_url()
    context_elements = [{"type": "mrkdwn", "text": "*Status:* `completed`"}]
    if run_url:
        context_elements.append({"type": "mrkdwn", "text": f"<{run_url}|Open Prefect run>"})

    blocks = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*{workflow_name} PDF available*"},
        },
        {"type": "context", "elements": context_elements},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*PDF:* <{console_url}|Open PDF in GCS>\n"
                    f"*GCS URI:* `{_escape_mrkdwn(gs_uri)}`\n"
                    f"*Sampled queries:* `{summary['sampled_query_count']}`\n"
                    f"*Total queries:* `{summary['total_query_count']}`"
                ),
            },
        },
    ]
    client = SlackWebhookClient(
        username="ml-workflows",
        icon_url=workflow_icon_url(workflow_name),
    )
    client.post(f"{workflow_name} PDF available: {gs_uri}", blocks=blocks)


@task(log_prints=True)
def download_artifact(bucket: str, blob_path: str) -> str:
    print(f"Downloading gs://{bucket}/{blob_path}")
    return download_blob(bucket, blob_path, local_dir="/tmp")


@task(log_prints=True)
def build_retrieval_visualization_pdf(
    documents_path: str,
    per_query_results_path: str,
    output_path: str,
    sample_count: int,
    sample_seed: int,
) -> dict:
    documents = _read_jsonl(documents_path)
    per_query_results = _read_jsonl(per_query_results_path)

    documents_by_id = {str(doc["doc_id"]): doc for doc in documents}
    print(
        f"Building retrieval visualization PDF: "
        f"documents={len(documents)}, queries={len(per_query_results)}, sample_count={sample_count}"
    )
    summary = write_retrieval_visualization_pdf(
        output_path=Path(output_path),
        per_query_results=per_query_results,
        documents_by_id=documents_by_id,
        sample_count=sample_count,
        sample_seed=sample_seed,
    )
    print(
        f"Built retrieval visualization PDF: "
        f"sampled={summary['sampled_query_count']}/{summary['total_query_count']} queries"
    )
    return summary


@task(log_prints=True)
def upload_retrieval_pdf(local_path: str, bucket: str, blob_path: str) -> None:
    print(f"Uploading retrieval visualization PDF to gs://{bucket}/{blob_path}")
    upload_blob(local_path, bucket, blob_path)


@slack_notified_flow(workflow_name="visualize-embedding-retrieval", log_prints=True)
def visualize_embedding_retrieval_flow(
    eval_dataset_bucket: str,
    eval_dataset_prefix: str,
    eval_report_bucket: str,
    eval_report_prefix: str,
    dest_bucket: str,
    dest_blob: str,
    sample_count: int = 20,
    sample_seed: int = 0,
    workflow_name: str = "visualize-embedding-retrieval",
) -> None:
    documents_path = download_artifact(eval_dataset_bucket, f"{eval_dataset_prefix.rstrip('/')}/documents.jsonl")
    per_query_results_path = download_artifact(eval_report_bucket, f"{eval_report_prefix.rstrip('/')}/per_query_results.jsonl")

    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pdf", dir="/tmp") as tmp:
        output_path = tmp.name

    try:
        summary = build_retrieval_visualization_pdf(
            documents_path=documents_path,
            per_query_results_path=per_query_results_path,
            output_path=output_path,
            sample_count=sample_count,
            sample_seed=sample_seed,
        )
        upload_retrieval_pdf(output_path, dest_bucket, dest_blob)
        _notify_pdf_available(dest_bucket, dest_blob, summary, workflow_name)
    finally:
        Path(documents_path).unlink(missing_ok=True)
        Path(per_query_results_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)
