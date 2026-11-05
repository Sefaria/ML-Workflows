import shutil
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from prefect import task

from embeddings.steps.training.model_cache import cache_huggingface_model
from roshei_tevot.steps.data import DATASET_REPO
from roshei_tevot.steps.index import ExpansionIndex
from roshei_tevot.steps.model import load_for_inference
from utils.gcs import download_directory, upload_directory
from utils.slack import slack_notified_flow


WORKFLOW_NAME = "build-expansion-index"
LEXICON_FILE = "lexicon/surface_index.jsonl"


@task(log_prints=True)
def build_and_upload(model_dir: str, lexicon_path: str, bucket: str, prefix: str, work_dir: str) -> dict:
    import torch

    model, tokenizer, _ = load_for_inference(model_dir, "cuda" if torch.cuda.is_available() else "cpu")
    index = ExpansionIndex.build(lexicon_path, model, tokenizer, "cuda" if torch.cuda.is_available() else "cpu")
    out = str(Path(work_dir) / "index")
    index.save(out)
    upload_directory(out, bucket, prefix)
    print(f"Indexed {len(index.expansions)} expansions for {len(index.by_surface)} surfaces")
    return {"expansions": len(index.expansions), "surfaces": len(index.by_surface), "uri": f"gs://{bucket}/{prefix}"}


@slack_notified_flow(workflow_name=WORKFLOW_NAME, log_prints=True)
def build_expansion_index_flow(
    model_bucket: str,
    model_prefix: str,
    index_prefix: str,
    dataset_repo: str = DATASET_REPO,
    lexicon_file: str = LEXICON_FILE,
) -> None:
    """Embed every lexicon expansion with a trained model and publish the index next to it (RAM-sized: under 100 MB in fp16)."""
    work_dir = tempfile.mkdtemp(dir="/tmp", prefix="expansion-index-")
    try:
        model_dir = download_directory(model_bucket, model_prefix, str(Path(work_dir) / "model"))
        lexicon_path = hf_hub_download(dataset_repo, lexicon_file, repo_type="dataset")
        build_and_upload(model_dir, lexicon_path, model_bucket, index_prefix, work_dir)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
