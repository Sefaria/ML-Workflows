import torch
from ..attention import resolve_attention_implementation
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer


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


def init_extended_embeddings(model, new_vocab_size: int, init_map: dict[str, list[int]]) -> int:
    """Grow the embedding matrix and start each new token at the mean of its old subtokens.

    Returns the number of rows added. The decoder is tied to the embeddings, so the output
    side grows with them; new decoder bias entries start at zero.
    """
    old_size = model.get_input_embeddings().weight.shape[0]
    if new_vocab_size <= old_size:
        return 0
    model.resize_token_embeddings(new_vocab_size, mean_resizing=False)
    embeddings = model.get_input_embeddings().weight
    with torch.no_grad():
        for new_id, old_ids in init_map.items():
            embeddings[int(new_id)] = embeddings[torch.tensor(old_ids)].mean(dim=0)
    return new_vocab_size - old_size
