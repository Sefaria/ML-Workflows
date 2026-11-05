import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer

from .encoding import add_markers, marker_ids

CONFIG_FILE = "siamese_config.json"
POOLINGS = ("span", "cls", "mean")


class SiameseExpander(nn.Module):
    """One shared encoder for contexts and expansions.

    Queries are pooled over the marked abbreviation (`span`, markers included), by default; candidates are mean pooled.
    Outputs are L2-normalised, so a dot product is a cosine similarity.
    """

    def __init__(self, encoder, pooling: str = "span", init_scale: float = 20.0, max_scale: float = 100.0):
        super().__init__()
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}")
        self.encoder = encoder
        self.pooling = pooling
        self.max_log_scale = math.log(max_scale)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(init_scale)))

    def _hidden(self, input_ids, attention_mask, token_type_ids):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state

    @staticmethod
    def _mean(hidden, mask):
        mask = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)

    def encode_queries(self, input_ids, attention_mask, token_type_ids, span_mask):
        hidden = self._hidden(input_ids, attention_mask, token_type_ids)
        if self.pooling == "span":
            pooled = self._mean(hidden, span_mask)
        elif self.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            pooled = self._mean(hidden, attention_mask)
        return F.normalize(pooled.float(), dim=-1)

    def encode_candidates(self, input_ids, attention_mask, token_type_ids):
        return F.normalize(self._mean(self._hidden(input_ids, attention_mask, token_type_ids), attention_mask).float(), dim=-1)

    def forward(self, queries: dict, candidates: dict, targets: torch.Tensor, valid: Optional[torch.Tensor] = None):
        """InfoNCE from each query to the shared candidate pool.

        `targets[i]` is the pool index of query i's expansion. The pool is deduplicated by text upstream, so the same expansion
        never appears twice as a negative. `valid` is an optional [queries, pool] mask of entries to exclude (extra false negatives).
        """
        q = self.encode_queries(**queries)
        c = self.encode_candidates(**candidates)
        scale = self.logit_scale.clamp(max=self.max_log_scale).exp()
        logits = scale * q @ c.T
        if valid is not None:
            logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        loss = F.cross_entropy(logits, targets)
        with torch.no_grad():
            accuracy = (logits.argmax(-1) == targets).float().mean()
        return {"loss": loss, "accuracy": accuracy}


def load_tokenizer(path: str):
    tokenizer = AutoTokenizer.from_pretrained(path)
    add_markers(tokenizer)
    return tokenizer


def build_model(base_model_path: str, pooling: str = "span"):
    """Base encoder plus marker tokens. New rows start at the mean of the quote and apostrophe embeddings."""
    tokenizer = load_tokenizer(base_model_path)
    encoder = AutoModel.from_pretrained(base_model_path, add_pooling_layer=False)
    old_size = encoder.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > old_size:
        encoder.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        embeddings = encoder.get_input_embeddings().weight
        anchors = [i for i in tokenizer.convert_tokens_to_ids(['"', "'"]) if i != tokenizer.unk_token_id]
        with torch.no_grad():
            start = embeddings[torch.tensor(anchors)].mean(0) if anchors else embeddings.mean(0)
            for token_id in marker_ids(tokenizer):
                embeddings[token_id] = start
    return SiameseExpander(encoder, pooling), tokenizer


def save_model(model: SiameseExpander, tokenizer, directory: str, extra: Optional[dict] = None) -> None:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    config = {"pooling": model.pooling, "logit_scale": float(model.logit_scale.exp()), **(extra or {})}
    (out / CONFIG_FILE).write_text(json.dumps(config, indent=2))


def load_for_inference(path: str, device: str = "cuda"):
    """Load a trained directory, or a plain BEREL directory as an untrained baseline (markers are added, pooling is `span`)."""
    config_file = Path(path) / CONFIG_FILE
    config = json.loads(config_file.read_text()) if config_file.exists() else {"pooling": "span"}
    model, tokenizer = build_model(path, config["pooling"])
    model.to(device).eval()
    return model, tokenizer, config
