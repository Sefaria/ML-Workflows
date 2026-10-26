import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..data.masking import mask_input_ids


DEFAULT_FILL_MASK_PROMPTS = (
    "Rashi was a medieval [MASK] who wrote a commentary on the Torah.",
    "The Talmud records a debate between Hillel and [MASK] about the proper way to teach.",
    "On Passover, Jews eat [MASK] to remember the exodus from Egypt.",
    "The Rambam wrote the Mishneh Torah, a comprehensive code of Jewish [MASK].",
    "And God said to Moses: I am that I [MASK].",
)


class StaticMaskedDataset(Dataset):
    """Pre-masked windows with a fixed seed so eval numbers are comparable across runs and models."""

    def __init__(self, windows: np.ndarray, tokenizer, mlm_probability: float, seed: int):
        self.input_ids, self.labels = mask_input_ids(
            torch.from_numpy(windows.astype(np.int64)),
            special_token_ids=tokenizer.all_special_ids,
            mask_token_id=tokenizer.mask_token_id,
            vocab_size=len(tokenizer),
            mlm_probability=mlm_probability,
            generator=torch.Generator().manual_seed(seed),
        )

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.input_ids[index], "labels": self.labels[index]}


@torch.no_grad()
def evaluate_masked_lm(
    model,
    dataset: StaticMaskedDataset,
    batch_size: int,
    device: Optional[str] = None,
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    loss_sum = 0.0
    correct = 0
    masked_tokens = 0
    for start in range(0, len(dataset), batch_size):
        input_ids = dataset.input_ids[start : start + batch_size].to(device)
        labels = dataset.labels[start : start + batch_size].to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(input_ids=input_ids).logits
        selected = labels != -100
        selected_logits = logits[selected].float()
        selected_labels = labels[selected]
        loss_sum += float(F.cross_entropy(selected_logits, selected_labels, reduction="sum"))
        correct += int((selected_logits.argmax(dim=-1) == selected_labels).sum())
        masked_tokens += int(selected.sum())

    if masked_tokens == 0:
        raise ValueError("Evaluation dataset has no masked tokens.")
    loss = loss_sum / masked_tokens
    return {
        "loss": loss,
        "perplexity": math.exp(min(loss, 50.0)),
        "masked_accuracy": correct / masked_tokens,
        "masked_tokens": masked_tokens,
        "sequences": len(dataset),
    }


@torch.no_grad()
def fill_mask_samples(
    model,
    tokenizer,
    prompts: tuple[str, ...] = DEFAULT_FILL_MASK_PROMPTS,
    top_k: int = 5,
    device: Optional[str] = None,
) -> list[dict]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    results = []
    for prompt in prompts:
        encoded = tokenizer(prompt.replace("[MASK]", tokenizer.mask_token), return_tensors="pt").to(device)
        mask_positions = (encoded["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        logits = model(**encoded).logits[0, mask_positions[0]]
        top_ids = logits.topk(top_k).indices.tolist()
        results.append({"prompt": prompt, "predictions": [tokenizer.decode([token_id]).strip() for token_id in top_ids]})
    return results
