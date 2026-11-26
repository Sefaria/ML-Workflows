"""Lemmatize the allusion-detection source corpus (Tanakh/Mishnah/Talmud Bavli
segments) and citing corpus (translation-scope works, chunked) with Shoshan,
recording the full top-K lemma candidates per word (see
app/allusion_detection/steps/lemmatize.py for why top-K rather than top-1).

Input data (tokenized text, via ParallelMatcher's tokenizer) and the Shoshan model
weights are pre-staged to GCS -- this flow does not fetch from Sefaria's database
or make bulk requests against Sefaria's public API, both of which it has no
business doing from inside a cluster job. See
Sefaria/translation-engine, experiments/text_matching for how the input data was
produced.
"""
import os
import pickle
import tempfile
from pathlib import Path

from prefect import task

from allusion_detection.steps.lemmatize import chunk_citing_corpus, lemmatize_units, load_shoshan_model
from utils.gcs import blob_exists, download_blob, download_directory, upload_blob
from utils.slack import SlackProgressReporter, slack_notified_flow


@task(log_prints=True)
def download_pickle(bucket: str, blob_path: str):
    local_path = download_blob(bucket, blob_path)
    with open(local_path, "rb") as f:
        data = pickle.load(f)
    os.remove(local_path)
    return data


@task(log_prints=True)
def download_shoshan(bucket: str, model_prefix: str, bank_prefix: str, local_dir: str):
    model_dir = str(Path(local_dir) / "model")
    bank_dir = str(Path(local_dir) / "bank")
    download_directory(bucket, model_prefix, model_dir)
    download_directory(bucket, bank_prefix, bank_dir)
    return model_dir, bank_dir


def _load_checkpoint(bucket: str, checkpoint_blob: str):
    if not blob_exists(bucket, checkpoint_blob):
        return [], 0
    local_path = download_blob(bucket, checkpoint_blob)
    with open(local_path, "rb") as f:
        ckpt = pickle.load(f)
    os.remove(local_path)
    print(f"Resuming from checkpoint: {ckpt['done_items']} word-items already done")
    return ckpt["all_topk"], ckpt["done_items"]


def _save_checkpoint(bucket: str, checkpoint_blob: str, all_topk: list, done_items: int):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        pickle.dump({"all_topk": all_topk, "done_items": done_items}, tmp)
        tmp_path = tmp.name
    upload_blob(tmp_path, bucket, checkpoint_blob)
    os.remove(tmp_path)


@task(log_prints=True)
def lemmatize_and_upload(
    model,
    device: str,
    units: list,
    *,
    label: str,
    bucket: str,
    checkpoint_blob: str,
    dest_blob: str,
    workflow_name: str,
):
    # GPU memory can comfortably hold much larger batches than the batch=256 default
    # (tuned for CPU); bigger batches mean fewer, better-utilized forward passes.
    batch = 1024 if device == "cuda" else 256
    resume_all_topk, resume_done_items = _load_checkpoint(bucket, checkpoint_blob)
    items_total = sum(len(words) for _, words in units)
    reporter = SlackProgressReporter(workflow_name=workflow_name, total_units=items_total, unit_label="words")
    reporter.notify_start({"stage": label, "units": len(units), "device": device, "batch": batch})

    def on_progress(done, total):
        reporter.notify_progress_if_due(done, {"stage": label})

    def on_checkpoint(all_topk, done_items):
        _save_checkpoint(bucket, checkpoint_blob, all_topk, done_items)

    try:
        out = lemmatize_units(
            model,
            units,
            batch=batch,
            on_progress=on_progress,
            resume_all_topk=resume_all_topk,
            resume_done_items=resume_done_items,
            on_checkpoint=on_checkpoint,
        )
    except Exception as exc:
        reporter.notify_failure(exc, {"stage": label})
        raise

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        pickle.dump(out, tmp)
        tmp_path = tmp.name
    upload_blob(tmp_path, bucket, dest_blob)
    os.remove(tmp_path)
    reporter.notify_success({"stage": label, "dest": f"gs://{bucket}/{dest_blob}"})
    return dest_blob


@slack_notified_flow(workflow_name="lemmatize-corpus", log_prints=True)
def lemmatize_corpus_flow(
    data_bucket: str = "sefaria-ml-models",
    source_segments_blob: str = "allusion-detection/data/source_segments_raw.pkl",
    citing_tokens_blob: str = "allusion-detection/data/citing_corpus_tokens.pkl",
    model_bucket: str = "sefaria-ml-models",
    model_prefix: str = "allusion-detection/models/shoshan/model",
    bank_prefix: str = "allusion-detection/models/shoshan/bank",
    dest_bucket: str = "sefaria-ml-models",
    source_dest_blob: str = "allusion-detection/output/source_lemmas.pkl",
    citing_dest_blob: str = "allusion-detection/output/citing_lemmas_chunks.pkl",
    citing_chunk_words: int = 15,
    run_source: bool = True,
    run_citing: bool = True,
) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_dir, bank_dir = download_shoshan(model_bucket, model_prefix, bank_prefix, tmp_dir)
        model, device = load_shoshan_model(model_dir, bank_dir)

        if run_source:
            source_segments = download_pickle(data_bucket, source_segments_blob)
            lemmatize_and_upload(
                model,
                device,
                source_segments,
                label="source",
                bucket=dest_bucket,
                checkpoint_blob="allusion-detection/output/.checkpoints/source_lemmas.checkpoint.pkl",
                dest_blob=source_dest_blob,
                workflow_name="lemmatize-corpus",
            )

        if run_citing:
            citing_corpus = download_pickle(data_bucket, citing_tokens_blob)
            citing_units = chunk_citing_corpus(citing_corpus, chunk=citing_chunk_words)
            lemmatize_and_upload(
                model,
                device,
                citing_units,
                label="citing",
                bucket=dest_bucket,
                checkpoint_blob="allusion-detection/output/.checkpoints/citing_lemmas.checkpoint.pkl",
                dest_blob=citing_dest_blob,
                workflow_name="lemmatize-corpus",
            )
