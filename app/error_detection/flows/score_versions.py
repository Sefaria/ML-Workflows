import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from prefect import task

from error_detection.steps.model import load_for_inference
from error_detection.steps.scoring import aggregate, load_segments, score_segments, to_csv
from utils.gcs import download_directory, write_blob_text
from utils.slack import notify_workflow_event, slack_notified_flow


WORKFLOW_NAME = "score-versions"
SAMPLE_REPO = "gs://sefaria-ml-models/error-detection/data/version-sample"


@task(log_prints=True)
def score(model_bucket: str, model_prefix: str, sample_repo: str, work_dir: str, threshold: Optional[float], max_segments: Optional[int],
          batch_size: int) -> dict:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = download_directory(model_bucket, model_prefix, str(Path(work_dir) / "model"))
    model, tokenizer, config = load_for_inference(model_dir, device)
    chosen = threshold if threshold is not None else config.get("validation_threshold")
    if chosen is None:
        raise ValueError("No threshold given and the model has no validation_threshold in error_detection_config.json.")
    rows = load_segments(sample_repo, max_segments)
    counts, top = score_segments(model, tokenizer, rows, device, config.get("max_length", 256), chosen, batch_size)
    print(f"Scored {len(counts)} segments, {sum(c[0] for c in counts.values())} words at threshold {chosen:.4f}")
    return {"threshold": chosen, "versions": aggregate(rows, counts, ("version",)), "books": aggregate(rows, counts, ("version", "book")),
            "examples": top.rows()}


@slack_notified_flow(workflow_name=WORKFLOW_NAME, log_prints=True)
def score_versions_flow(
    model_prefix: str,
    output_prefix: str,
    model_bucket: str = "sefaria-ml-models",
    sample_repo: str = SAMPLE_REPO,
    threshold: Optional[float] = None,
    max_segments: Optional[int] = None,
    batch_size: int = 128,
) -> None:
    """Flagged words per 1,000 for each version (and version/book) in the fixed segment sample, with 95% bootstrap intervals over segments,
    plus the top flagged words per version for reading by hand. The threshold defaults to the one the model chose on validation."""
    work_dir = tempfile.mkdtemp(dir="/tmp", prefix="error-detection-score-")
    try:
        result = score(model_bucket, model_prefix, sample_repo, work_dir, threshold, max_segments, batch_size)
        prefix = output_prefix.rstrip("/")
        write_blob_text(model_bucket, f"{prefix}/versions.csv", to_csv(result["versions"]))
        write_blob_text(model_bucket, f"{prefix}/books.csv", to_csv(result["books"]))
        write_blob_text(model_bucket, f"{prefix}/examples.json", json.dumps(result["examples"], ensure_ascii=False, indent=1))
        write_blob_text(model_bucket, f"{prefix}/meta.json", json.dumps({"model_prefix": model_prefix, "threshold": result["threshold"],
                                                                        "sample_repo": sample_repo, "versions": len(result["versions"])}, indent=1))
        notify_workflow_event(workflow_name=WORKFLOW_NAME, title=f"{WORKFLOW_NAME} completed", status="completed",
                              details={"output": f"gs://{model_bucket}/{prefix}", "versions": len(result["versions"])})
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
