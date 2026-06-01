import json
import tempfile
from pathlib import Path

from prefect import task

from embeddings.steps.query_generation.report import write_query_dataset_pdf
from utils.gcs import download_blob, upload_blob
from utils.slack import slack_notified_flow


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@task(log_prints=True)
def download_dataset_artifact(bucket: str, blob_path: str) -> str:
    print(f"Downloading gs://{bucket}/{blob_path}")
    return download_blob(bucket, blob_path, local_dir="/tmp")


@task(log_prints=True)
def build_query_dataset_visualization_pdf(
    documents_path: str,
    queries_path: str,
    qrels_path: str,
    output_path: str,
    sample_count: int,
    sample_seed: int,
) -> dict:
    documents = _read_jsonl(documents_path)
    queries = _read_jsonl(queries_path)
    qrels = _read_jsonl(qrels_path)

    qrels_by_query_id = {str(qrel["query_id"]): qrel for qrel in qrels}
    queries_by_doc_id: dict[str, list[dict]] = {}
    for query in queries:
        query_id = str(query["query_id"])
        qrel = qrels_by_query_id.get(query_id)
        if not qrel:
            continue
        doc_id = str(qrel["doc_id"])
        enriched_query = dict(query)
        enriched_query["reason"] = qrel.get("reason", "")
        queries_by_doc_id.setdefault(doc_id, []).append(enriched_query)

    selected_doc_ids = write_query_dataset_pdf(
        Path(output_path),
        documents,
        queries_by_doc_id,
        sample_count,
        sample_seed,
    )
    print(f"Built query dataset visualization PDF for {len(selected_doc_ids)} sampled document(s)")
    return {
        "sample_count": len(selected_doc_ids),
        "sample_seed": sample_seed,
        "selected_doc_ids": selected_doc_ids,
        "output_path": output_path,
    }


@task(log_prints=True)
def upload_visualization_pdf(local_path: str, bucket: str, blob_path: str) -> None:
    print(f"Uploading visualization PDF to gs://{bucket}/{blob_path}")
    upload_blob(local_path, bucket, blob_path)


@slack_notified_flow(workflow_name="visualize-query-dataset", log_prints=True)
def visualize_query_dataset_flow(
    source_bucket: str,
    source_prefix: str,
    dest_bucket: str,
    dest_blob: str,
    sample_count: int = 10,
    sample_seed: int = 0,
) -> None:
    documents_path = download_dataset_artifact(source_bucket, f"{source_prefix}/documents.jsonl")
    queries_path = download_dataset_artifact(source_bucket, f"{source_prefix}/queries.jsonl")
    qrels_path = download_dataset_artifact(source_bucket, f"{source_prefix}/qrels.jsonl")
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pdf", dir="/tmp") as tmp:
        output_path = tmp.name

    try:
        build_query_dataset_visualization_pdf(
            documents_path,
            queries_path,
            qrels_path,
            output_path,
            sample_count,
            sample_seed,
        )
        upload_visualization_pdf(output_path, dest_bucket, dest_blob)
    finally:
        Path(documents_path).unlink(missing_ok=True)
        Path(queries_path).unlink(missing_ok=True)
        Path(qrels_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)
