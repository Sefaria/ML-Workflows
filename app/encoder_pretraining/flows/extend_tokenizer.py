import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from prefect import task
from transformers import AutoTokenizer

from embeddings.steps.training.model_cache import cache_huggingface_model
from encoder_pretraining.steps.data import iter_documents
from encoder_pretraining.steps.tokenizer import (
    assert_special_tokens_unchanged,
    build_extended_tokenizer,
    count_hebrew_words,
    hebrew_token_usage,
    learn_hebrew_merges,
    new_token_init_map,
    save_extended_tokenizer,
)
from encoder_pretraining.steps.tokenizer.extend import NEW_TOKEN_INIT_FILE
from utils.gcs import download_blob, upload_directory
from utils.slack import notify_workflow_event, slack_notified_flow


WORKFLOW_NAME = "extend-tokenizer"
ENGLISH_CHECK_DOCUMENTS = 600
ENGLISH_CHECK_CHARACTERS = 6000


@task(log_prints=True)
def cache_base_model(model_repo_id: str, target_dir: str, hub_cache_dir: str, revision: Optional[str]) -> str:
    cache_huggingface_model(
        model_repo_id=model_repo_id,
        target_dir=target_dir,
        hub_cache_dir=hub_cache_dir,
        revision=revision,
        force_download=False,
    )
    return target_dir


@task(log_prints=True)
def download_training_data(bucket: str, blob_path: str) -> str:
    print(f"Downloading training data from gs://{bucket}/{blob_path}")
    return download_blob(bucket, blob_path, local_dir="/tmp")


@task(log_prints=True)
def count_words(documents_path: str) -> dict:
    train = count_hebrew_words(iter_documents(documents_path), "train")
    evaluation = count_hebrew_words(iter_documents(documents_path), "eval")
    print(f"Hebrew words: train {sum(train.values()):,} ({len(train):,} types), eval {sum(evaluation.values()):,}")
    return {"train": train, "eval": evaluation}


@task(log_prints=True)
def check_english_unchanged(documents_path: str, base_tokenizer, extended_tokenizer) -> dict:
    """Both tokenizers must produce identical ids for English text, or the base model's knowledge does not carry over."""
    texts = []
    for document in iter_documents(documents_path):
        # Screen the exact text that gets tokenized: Hebrew anywhere in it legitimately changes the ids.
        sample = document["text"][:ENGLISH_CHECK_CHARACTERS]
        if document["split"] == "train" and not any("\u0590" <= ch <= "\u05FF" for ch in sample):
            texts.append(sample)
        if len(texts) >= ENGLISH_CHECK_DOCUMENTS:
            break
    base_ids = base_tokenizer(texts, add_special_tokens=False)["input_ids"]
    extended_ids = extended_tokenizer(texts, add_special_tokens=False)["input_ids"]
    identical = sum(a == b for a, b in zip(base_ids, extended_ids))
    print(f"English documents tokenized identically: {identical}/{len(texts)}")
    return {"documents": len(texts), "identical": identical}


@slack_notified_flow(workflow_name=WORKFLOW_NAME, log_prints=True)
def extend_tokenizer_flow(
    source_bucket: str,
    source_blob: str,
    dest_bucket: str,
    dest_prefix: str,
    model_repo_id: str = "answerdotai/ModernBERT-base",
    revision: Optional[str] = None,
    cache_root: str = "/cache/huggingface",
    num_new_tokens: int = 6000,
    num_candidate_merges: int = 30000,
) -> None:
    """Add Hebrew tokens to ModernBERT's tokenizer, learned from the training corpus.

    Vowel marks are stripped by the tokenizer's normalizer, because they split a word into
    pieces that no merge can span. The new merges are learned on the training split only.
    """
    safe_model_name = model_repo_id.replace("/", "__")
    base_model_dir = str(Path(cache_root) / "models" / safe_model_name)
    hub_cache_dir = str(Path(cache_root) / "hub")
    output_dir = tempfile.mkdtemp(dir="/tmp", prefix="extended-tokenizer-")
    started_at = datetime.now(timezone.utc).isoformat()

    documents_path = None
    try:
        cache_base_model(model_repo_id, base_model_dir, hub_cache_dir, revision)
        documents_path = download_training_data(source_bucket, source_blob)
        counts = count_words(documents_path)

        base_tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
        base_json = json.loads(base_tokenizer.backend_tokenizer.to_str())
        merges = learn_hebrew_merges(counts["train"], num_candidate_merges)
        tokenizer_json, new_ids = build_extended_tokenizer(base_json, merges, num_new_tokens, first_new_id=len(base_tokenizer))
        extended = save_extended_tokenizer(tokenizer_json, base_tokenizer, output_dir)
        assert_special_tokens_unchanged(base_tokenizer, AutoTokenizer.from_pretrained(output_dir))

        english = check_english_unchanged(documents_path, base_tokenizer, extended)
        if english["identical"] != english["documents"]:
            raise RuntimeError(f"English tokenization changed for {english['documents'] - english['identical']} documents; refusing to publish this tokenizer.")

        usage = hebrew_token_usage(extended, counts["train"], counts["eval"], new_ids)
        (Path(output_dir) / NEW_TOKEN_INIT_FILE).write_text(json.dumps(new_token_init_map(base_tokenizer, extended, new_ids)))
        report = {
            "status": "success",
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "source": f"gs://{source_bucket}/{source_blob}",
            "base_model": model_repo_id,
            "base_vocab_size": len(base_tokenizer),
            "extended_vocab_size": len(extended),
            "english_check": english,
            "usage": usage,
        }
        (Path(output_dir) / "extension_report.json").write_text(json.dumps(report, indent=2))
        upload_directory(output_dir, dest_bucket, dest_prefix)
        notify_workflow_event(
            workflow_name=WORKFLOW_NAME,
            title=f"{WORKFLOW_NAME} completed",
            status="completed",
            details={
                "tokenizer_uri": f"gs://{dest_bucket}/{dest_prefix}",
                "new_tokens": usage["new_tokens"],
                "tokens_per_hebrew_word": round(usage["train_tokens_per_hebrew_word"], 2),
                "new_token_uses_p5": usage["new_token_uses_per_epoch_p5"],
                "new_tokens_used_under_100": usage["new_tokens_used_under_100"],
            },
        )
    finally:
        if documents_path:
            Path(documents_path).unlink(missing_ok=True)
        shutil.rmtree(output_dir, ignore_errors=True)
