import os
import tempfile

import pandas as pd
import torch
from prefect import flow, task
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

from utils.gcs import download_blob, upload_directory


@task(log_prints=True)
def download_dataset(bucket: str, blob_path: str) -> str:
    print(f"Downloading dataset from gs://{bucket}/{blob_path}")
    return download_blob(bucket, blob_path)


@task(log_prints=True)
def train_model(dataset_path: str) -> str:
    """Fine-tune a sentence embedding model on the normalized dataset.
    TODO: customize model name, loss, and training args for your use case.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    df = pd.read_json(dataset_path, lines=True)

    # TODO: construct InputExamples from your dataset columns
    # Example assumes columns: text1, text2, label (cosine similarity score 0-1)
    examples = [
        InputExample(texts=[row["text1"], row["text2"]], label=float(row["label"]))
        for _, row in df.iterrows()
    ]

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=100,
        output_path=None,
    )

    output_dir = tempfile.mkdtemp()
    model.save(output_dir)
    print(f"Model saved to {output_dir}")
    return output_dir


@task(log_prints=True)
def save_model_to_gcs(model_dir: str, bucket: str, gcs_prefix: str) -> None:
    print(f"Uploading model to gs://{bucket}/{gcs_prefix}")
    upload_directory(model_dir, bucket, gcs_prefix)


@flow(log_prints=True)
def train_embeddings_flow(
    source_bucket: str,
    source_blob: str,
    model_bucket: str,
    model_prefix: str,
) -> None:
    dataset_path = download_dataset(source_bucket, source_blob)
    model_dir = train_model(dataset_path)
    save_model_to_gcs(model_dir, model_bucket, model_prefix)
    os.remove(dataset_path)
