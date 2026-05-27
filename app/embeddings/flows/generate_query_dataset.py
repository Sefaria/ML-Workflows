import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from prefect import flow, task

from embeddings.steps.query_generation import QueryGenerationConfig, generate_queries_and_qrels
from utils.gcs import download_blob, upload_directory


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@task(log_prints=True)
def download_chunked_documents(bucket: str, blob_path: str) -> str:
    print(f"Downloading chunked documents from gs://{bucket}/{blob_path}")
    return download_blob(bucket, blob_path, local_dir="/tmp")


@task(log_prints=True)
def build_query_dataset(local_path: str, output_dir: str, model: str, max_workers: int) -> None:
    documents = _read_jsonl(local_path)
    config = QueryGenerationConfig(
        model=model,
        llm_max_workers=max_workers,
        verbose=True,
    )
    queries, qrels = generate_queries_and_qrels(documents, config)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("documents.jsonl", documents),
        ("queries.jsonl", queries),
        ("qrels.jsonl", qrels),
    ):
        with (output_root / filename).open("w") as fout:
            for row in rows:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "documents_count": len(documents),
        "queries_count": len(queries),
        "qrels_count": len(qrels),
        "query_types": list(config.query_types),
        "query_types_per_doc": config.query_types_per_doc,
        "queries_per_type_per_doc": config.queries_per_type_per_doc,
        "llm_max_workers": max_workers,
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Wrote dataset artifacts to {output_root}")


@task(log_prints=True)
def upload_query_dataset(local_dir: str, bucket: str, prefix: str) -> None:
    print(f"Uploading query dataset artifacts to gs://{bucket}/{prefix}")
    upload_directory(local_dir, bucket, prefix)


@flow(log_prints=True)
def generate_query_dataset_flow(
    source_bucket: str,
    source_blob: str,
    dest_bucket: str,
    dest_prefix: str,
    model: str = "claude-sonnet-4-6",
    max_workers: int = 4,
) -> None:
    source_local_path = download_chunked_documents(source_bucket, source_blob)
    output_dir = tempfile.mkdtemp(dir="/tmp")

    try:
        build_query_dataset(source_local_path, output_dir, model, max_workers)
        upload_query_dataset(output_dir, dest_bucket, dest_prefix)
    finally:
        Path(source_local_path).unlink(missing_ok=True)
        shutil.rmtree(output_dir, ignore_errors=True)
