import os

import pandas as pd
from prefect import flow, task

from utils.gcs import download_blob, upload_blob
from utils.slack import notify_workflow_started


@task(log_prints=True)
def download_from_gcs(bucket: str, blob_path: str) -> str:
    print(f"Downloading gs://{bucket}/{blob_path}")
    return download_blob(bucket, blob_path)


@task(log_prints=True)
def normalize_data(local_path: str) -> pd.DataFrame:
    """Load the raw data file and normalize it into a dataset.
    TODO: implement normalization logic specific to your data format.
    """
    df = pd.read_json(local_path, lines=True)
    # TODO: add normalization steps here
    print(f"Normalized {len(df)} rows")
    return df


@task(log_prints=True)
def upload_to_gcs(df: pd.DataFrame, bucket: str, blob_path: str) -> None:
    local_path = "/tmp/dataset.jsonl"
    df.to_json(local_path, orient="records", lines=True)
    print(f"Uploading dataset to gs://{bucket}/{blob_path}")
    upload_blob(local_path, bucket, blob_path)
    os.remove(local_path)


@flow(log_prints=True)
def create_dataset_flow(
    source_bucket: str,
    source_blob: str,
    dest_bucket: str,
    dest_blob: str,
) -> None:
    notify_workflow_started(
        "create-dataset",
        {
            "Source": f"gs://{source_bucket}/{source_blob}",
            "Destination": f"gs://{dest_bucket}/{dest_blob}",
        },
    )
    local_path = download_from_gcs(source_bucket, source_blob)
    df = normalize_data(local_path)
    upload_to_gcs(df, dest_bucket, dest_blob)
    os.remove(local_path)
