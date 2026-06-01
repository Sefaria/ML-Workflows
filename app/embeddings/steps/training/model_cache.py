import shutil
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, BertForMaskedLM


def directory_size_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def sample_files(path: Path, limit: int = 50) -> list[str]:
    files = sorted(str(file.relative_to(path)) for file in path.rglob("*") if file.is_file())
    return files[:limit]


def cache_huggingface_model(
    model_repo_id: str,
    target_dir: str,
    hub_cache_dir: str,
    revision: Optional[str],
    force_download: bool,
) -> dict:
    target = Path(target_dir)
    hub_cache = Path(hub_cache_dir)

    if force_download and target.exists():
        shutil.rmtree(target)

    target.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)

    resolved_path = snapshot_download(
        repo_id=model_repo_id,
        revision=revision,
        local_dir=str(target),
        cache_dir=str(hub_cache),
        force_download=force_download,
    )

    resolved = Path(resolved_path)
    return {
        "model_repo_id": model_repo_id,
        "requested_revision": revision,
        "force_download": force_download,
        "target_dir": str(target),
        "hub_cache_dir": str(hub_cache),
        "resolved_path": str(resolved),
        "exists": resolved.exists(),
        "size_bytes": directory_size_bytes(target),
        "sample_files": sample_files(target),
    }


def validate_masked_lm_model(model_path: str) -> dict:
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
