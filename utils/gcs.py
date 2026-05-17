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
