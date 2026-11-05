import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer

from .encoding import add_markers, marker_ids

CONFIG_FILE = "roshei_tevot_config.json"
HEAD_FILE = "cross_head.pt"
POOLINGS = ("span", "cls", "mean")
ARCHITECTURES = ("siamese", "cross")


def _mean(hidden, mask):
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)


class SiameseExpander(nn.Module):
    """One shared encoder for contexts and expansions.

    Queries are pooled over the marked abbreviation (`span`, markers included) by default; candidates are mean pooled. Outputs are
    L2-normalised, so a dot product is a cosine similarity.

    Loss: InfoNCE from each query to the whole batch pool, plus `within_weight` times cross-entropy restricted to the row's own
    candidates (its target and mined negatives). The second term is the choice inference makes; the first keeps the space organised.
    """

    architecture = "siamese"

    def __init__(self, encoder, pooling: str = "span", within_weight: float = 1.0, init_scale: float = 20.0, max_scale: float = 100.0):
        super().__init__()
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}")
        self.encoder = encoder
        self.pooling = pooling
        self.within_weight = within_weight
        self.max_log_scale = math.log(max_scale)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(init_scale)))

    def _hidden(self, input_ids, attention_mask, token_type_ids):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state

    def scale(self) -> torch.Tensor:
        return self.logit_scale.clamp(max=self.max_log_scale).exp()

    def encode_queries(self, input_ids, attention_mask, token_type_ids, span_mask):
        hidden = self._hidden(input_ids, attention_mask, token_type_ids)
        if self.pooling == "span":
            pooled = _mean(hidden, span_mask)
        elif self.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            pooled = _mean(hidden, attention_mask)
        return F.normalize(pooled.float(), dim=-1)

    def encode_candidates(self, input_ids, attention_mask, token_type_ids):
        return F.normalize(_mean(self._hidden(input_ids, attention_mask, token_type_ids), attention_mask).float(), dim=-1)

    def forward(self, queries: dict, candidates: dict, targets: torch.Tensor, own: Optional[torch.Tensor] = None):
        logits = self.scale() * self.encode_queries(**queries) @ self.encode_candidates(**candidates).T
        loss = F.cross_entropy(logits, targets)
        scored = logits
        if own is not None:
            scored = logits.masked_fill(~own, torch.finfo(logits.dtype).min)
            if self.within_weight:
                loss = loss + self.within_weight * F.cross_entropy(scored, targets)
        with torch.no_grad():
            accuracy = (scored.argmax(-1) == targets).float().mean()
        return {"loss": loss, "accuracy": accuracy}


class CrossExpander(nn.Module):
    """Reads the context and one candidate together; the score comes from the CLS vector concatenated with the marked-span mean.
    Loss: cross-entropy over each row's candidates, the target at position 0 of its group."""

    architecture = "cross"

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(2 * hidden, 1))

    def score(self, input_ids, attention_mask, token_type_ids, span_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
        return self.head(torch.cat([hidden[:, 0], _mean(hidden, span_mask)], dim=-1)).squeeze(-1).float()

    def forward(self, pairs: dict, group: torch.Tensor, rows: int):
        scores = self.score(**pairs)
        counts = torch.bincount(group, minlength=rows)
        width = int(counts.max())
        offsets = torch.cumsum(counts, 0) - counts
        column = torch.arange(len(group), device=group.device) - offsets[group]
        grid = torch.full((rows, width), torch.finfo(scores.dtype).min, device=scores.device, dtype=scores.dtype)
        grid[group, column] = scores
        targets = torch.zeros(rows, dtype=torch.long, device=scores.device)
        loss = F.cross_entropy(grid, targets)
        with torch.no_grad():
            accuracy = (grid.argmax(-1) == 0).float().mean()
        return {"loss": loss, "accuracy": accuracy}


def load_tokenizer(path: str):
    tokenizer = AutoTokenizer.from_pretrained(path)
    add_markers(tokenizer)
    return tokenizer


def _encoder_with_markers(base_model_path: str):
    """Base encoder plus marker tokens. New rows start at the mean of the quote and apostrophe embeddings."""
    tokenizer = load_tokenizer(base_model_path)
    encoder = AutoModel.from_pretrained(base_model_path, add_pooling_layer=False)
    if len(tokenizer) > encoder.get_input_embeddings().weight.shape[0]:
        encoder.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        embeddings = encoder.get_input_embeddings().weight
        anchors = [i for i in tokenizer.convert_tokens_to_ids(['"', "'"]) if i != tokenizer.unk_token_id]
        with torch.no_grad():
            start = embeddings[torch.tensor(anchors)].mean(0) if anchors else embeddings.mean(0)
            for token_id in marker_ids(tokenizer):
                embeddings[token_id] = start
    return encoder, tokenizer


def build_model(base_model_path: str, architecture: str = "siamese", pooling: str = "span", within_weight: float = 1.0):
    if architecture not in ARCHITECTURES:
        raise ValueError(f"architecture must be one of {ARCHITECTURES}")
    encoder, tokenizer = _encoder_with_markers(base_model_path)
    model = SiameseExpander(encoder, pooling, within_weight) if architecture == "siamese" else CrossExpander(encoder)
    return model, tokenizer


def save_model(model, tokenizer, directory: str, extra: Optional[dict] = None) -> None:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    config = {"architecture": model.architecture, **(extra or {})}
    if isinstance(model, SiameseExpander):
        config.update(pooling=model.pooling, logit_scale=float(model.scale()))
    else:
        torch.save(model.head.state_dict(), out / HEAD_FILE)
    (out / CONFIG_FILE).write_text(json.dumps(config, indent=2))


def load_for_inference(path: str, device: str = "cuda"):
    """Load a trained directory, or a plain BEREL directory as an untrained siamese baseline (markers added, `span` pooling)."""
    config_file = Path(path) / CONFIG_FILE
    config = json.loads(config_file.read_text()) if config_file.exists() else {"architecture": "siamese", "pooling": "span", "logit_scale": 20.0}
    model, tokenizer = build_model(path, config["architecture"], config.get("pooling", "span"))
    if isinstance(model, CrossExpander):
        model.head.load_state_dict(torch.load(Path(path) / HEAD_FILE, map_location="cpu"))
    model.to(device).eval()
    return model, tokenizer, config
