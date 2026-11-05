import shutil
import tempfile
from pathlib import Path

from prefect import task

from roshei_tevot.steps.data import GCS_DATASET, load_lexicon
from roshei_tevot.steps.index import ExpansionIndex
from roshei_tevot.steps.model import load_for_inference
from utils.gcs import download_directory, upload_directory
from utils.slack import slack_notified_flow


WORKFLOW_NAME = "build-expansion-index"


@task(log_prints=True)
def build_and_upload(model_dir: str, dataset_repo: str, bucket: str, prefix: str, work_dir: str) -> dict:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, config = load_for_inference(model_dir, device)
    if config["architecture"] != "siamese":
        raise ValueError("Only a siamese model has expansion embeddings to index; a cross-encoder scores candidates at query time.")
    index = ExpansionIndex.build(load_lexicon(dataset_repo), model, tokenizer, device)
    out = str(Path(work_dir) / "index")
    index.save(out)
    upload_directory(out, bucket, prefix)
    print(f"Indexed {len(index.expansions)} expansions for {len(index.lexicon.by_surface)} surfaces")
    return {"expansions": len(index.expansions), "surfaces": len(index.lexicon.by_surface), "uri": f"gs://{bucket}/{prefix}"}


@slack_notified_flow(workflow_name=WORKFLOW_NAME, log_prints=True)
def build_expansion_index_flow(model_bucket: str, model_prefix: str, index_prefix: str, dataset_repo: str = GCS_DATASET) -> None:
    """Embed every lexicon expansion with a trained siamese model and publish the index (under 100 MB in fp16, meant to be held in RAM)."""
    work_dir = tempfile.mkdtemp(dir="/tmp", prefix="expansion-index-")
    try:
        model_dir = download_directory(model_bucket, model_prefix, str(Path(work_dir) / "model"))
        build_and_upload(model_dir, dataset_repo, model_bucket, index_prefix, work_dir)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
