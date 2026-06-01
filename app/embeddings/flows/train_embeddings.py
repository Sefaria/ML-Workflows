import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download
from prefect import flow, task
from transformers import AutoTokenizer, BertForMaskedLM

from utils.gcs import upload_blob


def _directory_size_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _sample_files(path: Path, limit: int = 50) -> list[str]:
    files = sorted(str(file.relative_to(path)) for file in path.rglob("*") if file.is_file())
    return files[:limit]


@task(log_prints=True)
def cache_base_model(
    model_repo_id: str,
    target_dir: str,
    hub_cache_dir: str,
    revision: Optional[str],
    force_download: bool,
) -> dict:
    target = Path(target_dir)
    hub_cache = Path(hub_cache_dir)

    if force_download and target.exists():
        print(f"force_download=true; removing existing model directory {target}")
        shutil.rmtree(target)

    target.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {model_repo_id} revision={revision or 'default'} to {target}")
    resolved_path = snapshot_download(
        repo_id=model_repo_id,
        revision=revision,
        local_dir=str(target),
        cache_dir=str(hub_cache),
        force_download=force_download,
    )

    resolved = Path(resolved_path)
    report = {
        "model_repo_id": model_repo_id,
        "requested_revision": revision,
        "force_download": force_download,
        "target_dir": str(target),
        "hub_cache_dir": str(hub_cache),
        "resolved_path": str(resolved),
        "exists": resolved.exists(),
        "size_bytes": _directory_size_bytes(target),
        "sample_files": _sample_files(target),
    }
    print(f"Base model is available at {resolved}; size_bytes={report['size_bytes']}")
    return report


@task(log_prints=True)
def validate_base_model(model_path: str) -> dict:
    print(f"Validating model can be loaded from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = BertForMaskedLM.from_pretrained(model_path)
    model.eval()
    return {
        "model_path": model_path,
        "tokenizer_class": tokenizer.__class__.__name__,
        "model_class": model.__class__.__name__,
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0) or 0),
        "hidden_size": int(getattr(model.config, "hidden_size", 0) or 0),
        "num_hidden_layers": int(getattr(model.config, "num_hidden_layers", 0) or 0),
    }


@task(log_prints=True)
def upload_training_probe_report(report: dict, bucket: str, blob_path: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
        json.dump(report, tmp, ensure_ascii=False, indent=2)
        local_path = tmp.name

    try:
        print(f"Uploading training workflow probe report to gs://{bucket}/{blob_path}")
        upload_blob(local_path, bucket, blob_path)
    finally:
        Path(local_path).unlink(missing_ok=True)


@flow(log_prints=True)
def train_embeddings_flow(
    report_bucket: str,
    report_blob: str,
    model_repo_id: str = "dicta-il/BEREL_3.0",
    revision: Optional[str] = None,
    cache_root: str = "/cache/huggingface",
    force_download: bool = False,
    validate_model: bool = True,
) -> None:
    safe_model_name = model_repo_id.replace("/", "__")
    target_dir = str(Path(cache_root) / "models" / safe_model_name)
    hub_cache_dir = str(Path(cache_root) / "hub")

    started_at = datetime.now(timezone.utc).isoformat()
    cache_report = cache_base_model(
        model_repo_id=model_repo_id,
        target_dir=target_dir,
        hub_cache_dir=hub_cache_dir,
        revision=revision,
        force_download=force_download,
    )
    validation_report = validate_base_model(target_dir) if validate_model else None
    finished_at = datetime.now(timezone.utc).isoformat()

    report = {
        "status": "success",
        "workflow_stage": "base_model_cache_probe",
        "started_at": started_at,
        "finished_at": finished_at,
        "model_repo_id": model_repo_id,
        "revision": revision,
        "cache_root": cache_root,
        "target_dir": target_dir,
        "hub_cache_dir": hub_cache_dir,
        "force_download": force_download,
        "validate_model": validate_model,
        "cache": cache_report,
        "validation": validation_report,
        "environment": {
            "HF_HOME": os.getenv("HF_HOME"),
            "HF_HUB_CACHE": os.getenv("HF_HUB_CACHE"),
            "TRANSFORMERS_CACHE": os.getenv("TRANSFORMERS_CACHE"),
        },
    }
    upload_training_probe_report(report, report_bucket, report_blob)
