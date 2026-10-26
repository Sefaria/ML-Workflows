import importlib.util

import torch
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer


def resolve_attention_implementation(requested: str) -> str:
    """Pick flash attention when asked for (or "auto") and installed, otherwise SDPA."""
    if requested not in ("auto", "flash_attention_2", "sdpa"):
        raise ValueError(f"Unsupported attn_implementation: {requested}")
    flash_available = importlib.util.find_spec("flash_attn") is not None
    if requested == "flash_attention_2" and not flash_available:
        raise RuntimeError("attn_implementation=flash_attention_2 requested but flash_attn is not installed.")
    if requested == "sdpa" or not flash_available:
        return "sdpa"
    return "flash_attention_2"


def load_masked_lm(model_path: str, attn_implementation: str, reference_compile: bool):
    # Weights stay fp32; bf16 autocast in the Trainer keeps optimizer state in full precision.
    return AutoModelForMaskedLM.from_pretrained(
        model_path,
        attn_implementation=attn_implementation,
        reference_compile=reference_compile,
    )


def validate_modernbert(model_path: str) -> dict:
    config = AutoConfig.from_pretrained(model_path)
    if config.model_type != "modernbert":
        raise ValueError(f"Expected a modernbert model at {model_path}, found model_type={config.model_type}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return {
        "model_path": model_path,
        "model_type": config.model_type,
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab_size": len(tokenizer),
        "hidden_size": int(config.hidden_size),
        "num_hidden_layers": int(config.num_hidden_layers),
        "max_position_embeddings": int(config.max_position_embeddings),
    }


def require_bf16_gpu() -> dict:
    """Fail fast if the job did not land on a bf16-capable GPU (for example a leftover P100 node)."""
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device is available; refusing to pretrain on CPU.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"GPU {torch.cuda.get_device_name(0)} does not support bf16.")
    props = torch.cuda.get_device_properties(0)
    return {
        "device_count": torch.cuda.device_count(),
        "name": torch.cuda.get_device_name(0),
        "total_memory_bytes": int(props.total_memory),
        "compute_capability": f"{props.major}.{props.minor}",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
