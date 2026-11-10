import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer

from .data import IGNORE

CONFIG_FILE = "error_detection_config.json"
LABELS = {0: "ok", 1: "error"}


def build_model(base_model_path: str):
    """BEREL with a 2-way token head. Dicta's card warns BertTokenizer gives bad results with BEREL 3.0, so load the fast tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        base_model_path, num_labels=2, id2label=LABELS, label2id={v: k for k, v in LABELS.items()})
    return model, tokenizer


def weighted_token_loss(logits: torch.Tensor, labels: torch.Tensor, pos_weight: float) -> torch.Tensor:
    weight = torch.tensor([1.0, pos_weight], device=logits.device, dtype=torch.float32)
    return F.cross_entropy(logits.float().view(-1, logits.shape[-1]), labels.view(-1), weight=weight, ignore_index=IGNORE)


class WeightedTokenTrainer(Trainer):
    """Cross-entropy with the error class up-weighted (about 1.6% of labelled words are errors). Keeps the best weights by validation
    PR-AUC in CPU memory; after a restart the best is tracked again from the resumed point."""

    def __init__(self, *args, pos_weight: float = 5.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight
        self.best_metric: Optional[float] = None
        self.best_step: Optional[int] = None
        self._best_state: Optional[dict] = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = weighted_token_loss(outputs.logits, labels, self.pos_weight)
        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        score = metrics.get(f"{metric_key_prefix}_pr_auc")
        if score is not None and (self.best_metric is None or score > self.best_metric):
            self.best_metric, self.best_step = score, self.state.global_step
            self._best_state = {k: v.detach().to("cpu", copy=True) for k, v in self.model.state_dict().items()}
        return metrics

    def restore_best(self) -> bool:
        if self._best_state is None:
            return False
        self.model.load_state_dict(self._best_state)
        return True


def save_model(model, tokenizer, directory: str, extra: Optional[dict] = None) -> None:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    (out / CONFIG_FILE).write_text(json.dumps(extra or {}, indent=2, ensure_ascii=False))


def load_for_inference(path: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(path).to(device).eval()
    config_file = Path(path) / CONFIG_FILE
    config = json.loads(config_file.read_text()) if config_file.exists() else {}
    return model, tokenizer, config
