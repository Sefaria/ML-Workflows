import json
import shutil
import tempfile
from pathlib import Path

from prefect import task

from error_detection.steps.data import GCS_DATASET, chunk_rows, load_rows, select_fold
from error_detection.steps.evaluation import predict_words
from error_detection.steps.model import load_for_inference
from utils.gcs import download_directory, write_blob_text
from utils.slack import notify_workflow_event, slack_notified_flow


WORKFLOW_NAME = "score-rows"


@task(log_prints=True)
def score(model_bucket: str, model_prefix: str, dataset_repo: str, config: str, split: str, n_folds: int, fold: int, min_score: float,
          work_dir: str, batch_size: int) -> list[dict]:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, model_config = load_for_inference(download_directory(model_bucket, model_prefix, str(Path(work_dir) / "model")), device)
    rows = select_fold(load_rows(config, split, dataset_repo), n_folds, fold, held_out=True)
    by_id = {row["id"]: row for row in rows}
    chunks = chunk_rows(tokenizer, rows, model_config.get("max_length", 256))
    words = predict_words(model, tokenizer, chunks, device, batch_size)
    out = []
    for w in words:
        if w["score"] >= min_score:
            row_id = w["chunk"].rsplit("#", 1)[0]
            out.append({"row_id": row_id, "word_index": w["word_index"], "token": by_id[row_id]["tokens"][w["word_index"]],
                        "score": round(w["score"], 5), "label": w["label"], "in_segment": w["in_segment"]})
    print(f"{len(rows)} rows, {len(words)} words, {len(out)} at score >= {min_score} ({sum(1 for o in out if o['label'] == 0)} labelled ok)")
    return out


@slack_notified_flow(workflow_name=WORKFLOW_NAME, log_prints=True)
def score_rows_flow(
    model_prefix: str,
    output_blob: str,
    model_bucket: str = "sefaria-ml-models",
    dataset_repo: str = GCS_DATASET,
    config: str = "real",
    split: str = "test",
    n_folds: int = 0,
    fold: int = 0,
    min_score: float = 0.3,
    batch_size: int = 64,
) -> None:
    """Every word of a dataset split (or of one held-out fold of it) that a trained model scores at `min_score` or above, as JSONL:
    row_id, word_index, token, score, label, in_segment. Used to find uncorrected errors for judging: run a model on validation/test, or a
    model trained with n_folds/fold on the training fold it never saw."""
    work_dir = tempfile.mkdtemp(dir="/tmp", prefix="error-detection-rows-")
    try:
        words = score(model_bucket, model_prefix, dataset_repo, config, split, n_folds, fold, min_score, work_dir, batch_size)
        write_blob_text(model_bucket, output_blob, "".join(json.dumps(w, ensure_ascii=False) + "\n" for w in words))
        notify_workflow_event(workflow_name=WORKFLOW_NAME, title=f"{WORKFLOW_NAME} completed", status="completed",
                              details={"output": f"gs://{model_bucket}/{output_blob}", "words": len(words)})
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
