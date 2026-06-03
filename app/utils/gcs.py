from __future__ import annotations

import os
import tempfile
from pathlib import Path

from google.cloud import storage


def download_blob(bucket_name: str, blob_path: str, local_dir: str | None = None) -> str:
    """Download a GCS blob to a local temp file. Returns the local path."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    suffix = Path(blob_path).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=local_dir)
    blob.download_to_filename(tmp.name)
    return tmp.name


def upload_blob(local_path: str, bucket_name: str, blob_path: str) -> None:
    """Upload a local file to GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)


def upload_directory(local_dir: str, bucket_name: str, gcs_prefix: str) -> None:
    """Recursively upload a local directory to a GCS prefix."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for root, _, files in os.walk(local_dir):
        for fname in files:
            local_file = os.path.join(root, fname)
            relative = os.path.relpath(local_file, local_dir)
            blob_name = f"{gcs_prefix}/{relative}"
            bucket.blob(blob_name).upload_from_filename(local_file)


def download_directory(bucket_name: str, gcs_prefix: str, local_dir: str) -> str:
    """Download all blobs under a GCS prefix into local_dir. Returns local_dir."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    normalized_prefix = gcs_prefix.rstrip("/") + "/"
    downloaded = 0
    for blob in client.list_blobs(bucket, prefix=normalized_prefix):
        if blob.name.endswith("/"):
            continue
        relative = blob.name[len(normalized_prefix) :]
        local_path = Path(local_dir) / relative
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
        downloaded += 1
    if downloaded == 0:
        raise FileNotFoundError(f"No blobs found under gs://{bucket_name}/{normalized_prefix}")
    return local_dir


def delete_prefix(bucket_name: str, gcs_prefix: str) -> int:
    """Delete all blobs under a GCS prefix. Returns the number of deleted blobs."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    normalized_prefix = gcs_prefix.rstrip("/") + "/"
    deleted_count = 0
    for blob in client.list_blobs(bucket, prefix=normalized_prefix):
        blob.delete()
        deleted_count += 1
    return deleted_count
