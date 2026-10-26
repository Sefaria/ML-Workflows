import importlib.util


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
