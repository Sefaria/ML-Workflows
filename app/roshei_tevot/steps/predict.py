"""Run a trained roshei tevot model on text: find the abbreviations, offer each one's readings, rank them in context, group them by sense.

    from roshei_tevot.steps.predict import RosheiTevot
    rt = RosheiTevot.from_pretrained("Sefaria/berel-roshei-tevot-siamese")        # or a local folder with the same layout
    for token in rt.expand(text)["tokens"]:
        print(token["surface"], token["senses"][0]["text"], token["senses"][0]["probability"])

The model only RANKS the readings it is given, so the folder holds more than weights (what `publish_hf.py` assembles):
    config.json, model.safetensors, tokenizer files, roshei_tevot_config.json    the siamese model (`logit_scale`, `pooling`)
    index/lexicon.json, index/embeddings.npy                                      the lexicon (which readings a token can have, with library frequencies) and one embedding per reading
    senses.json                                                                   key -> sense index -> {label, texts, expansions}: readings of one sense (spelling variants) are one answer

What happens to a text, in order (each step is something the model does not do for itself):
  1. `find`: detect tokens that look like abbreviations (prefix letters allowed), clean quote marks, drop letter and divine names written with gershayim (nonabbrev.py).
  2. candidates: the lexicon's readings of the token (as written or behind a prefix, letters must fit), plus the number for a numeral (also behind a prefix, one-letter
     geresh numerals too), `סעיף קטן X` for `סק"X`, the library's most frequent completions of a truncated word (לכתחי'), and citation readings (`סקי"א`) except for a bare
     numeral or a token ending in a geresh (there they would win at p~1 with `קיצור` / `סימן י'`).
  3. query: the token wrapped in the `<abbr>` markers, 200 words either side (`CONTEXT_WORDS`), no title / category (empty `| `); the model pools over the marked span.
  4. score: softmax over `logit_scale * cosine + prior_weight * log(1 + frequency)`; the prior (off by default) is the lexicon's library frequency of the reading, which does not help (see DEFAULT_PRIOR_WEIGHT).
  5. senses: probabilities of readings of one sense class are summed and the classes ranked.
"""
import json
from pathlib import Path
from typing import Optional

import torch

from .citations import expand as citation_readings
from .encoding import collate_candidates, collate_queries, encode_candidate, encode_query
from .index import ExpansionIndex, number_reading, prior
from .nonabbrev import is_non_abbreviation
from .vendor.detect import detect, parses
from .vendor.hebrew import clean, numeral_value

CONTEXT_WORDS = 200
MAX_LENGTH = 512
DEFAULT_PRIOR_WEIGHT = 0.0      # the model alone. The evaluation gains about 1.5 points top-1 from a prior of 0.25-1.0 on TRAIN-SET counts of each reading; the lexicon frequency used here is the library count, and on 700 held-out val rows it LOWERED top-1 (92.4 at 0, 91.3 at 0.25, 89.6 at 0.5, 86.9 at 1.0)
CONFIG_FILE = "roshei_tevot_config.json"


class RosheiTevot:
    def __init__(self, model, tokenizer, config: dict, index: ExpansionIndex, senses: Optional[dict] = None,
                 prior_weight: float = DEFAULT_PRIOR_WEIGHT, device: str = "cpu"):
        self.model, self.tokenizer, self.index, self.senses = model.to(device).eval(), tokenizer, index, senses or {}
        self.pooling = config.get("pooling", "span")
        self.scale = float(config["logit_scale"])
        self.prior_weight = prior_weight
        self.device = device
        self._embeddings: dict[str, torch.Tensor] = {}           # readings the index lacks (citation readings), embedded on demand
        self._sense_of: dict[str, dict[str, int]] = {}

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu", prior_weight: float = DEFAULT_PRIOR_WEIGHT, threads: Optional[int] = None) -> "RosheiTevot":
        """`path`: a folder laid out as in the module docstring, or a Hugging Face repo id (downloaded with `huggingface_hub.snapshot_download`, needs a login for a private repo)."""
        from transformers import AutoModel, AutoTokenizer

        if threads:
            torch.set_num_threads(threads)
        folder = Path(path)
        if not folder.is_dir():
            from huggingface_hub import snapshot_download

            folder = Path(snapshot_download(path))
        config = json.loads((folder / CONFIG_FILE).read_text())
        if config.get("architecture", "siamese") != "siamese":
            raise ValueError("Only a siamese model can be run from an index of expansion embeddings.")
        model = AutoModel.from_pretrained(str(folder), add_pooling_layer=False)
        tokenizer = AutoTokenizer.from_pretrained(str(folder))
        index = ExpansionIndex.load(str(folder / "index"))
        senses_file = folder / "senses.json"
        senses = json.loads(senses_file.read_text()) if senses_file.exists() else {}
        return cls(model, tokenizer, config, index, senses, prior_weight, device)

    # ---- senses -----------------------------------------------------------------------------------------------------------------

    def _expansion_to_sense(self, key: str) -> dict[str, int]:
        if key not in self._sense_of:
            table: dict[str, int] = {}
            for idx, entry in self.senses.get(key, {}).items():
                for text in entry["expansions"] + entry["texts"]:
                    table.setdefault(text, int(idx))
            self._sense_of[key] = table
        return self._sense_of[key]

    def sense_for(self, surface: str, expansion: str):
        """(key, sense index) of the first reading key of this token whose sense table lists the expansion, else (None, None)."""
        for key in [surface] + [bare for _, bare in parses(surface)]:
            idx = self._expansion_to_sense(key).get(expansion)
            if idx is not None:
                return key, idx
        return None, None

    def _describe(self, surface: str, probs: dict[str, float], freqs: dict[str, int]) -> list[dict]:
        """Group scored readings into sense classes, ranked by summed probability."""
        groups: dict[tuple, dict] = {}
        for text, p in probs.items():
            key, idx = self.sense_for(surface, text)
            gid = (key, idx) if idx is not None else ("_single", text)
            g = groups.setdefault(gid, {"key": key, "sense": idx, "label": "", "readings": [], "probability": 0.0, "freq": 0})
            g["probability"] += p
            g["freq"] += freqs[text]
            g["readings"].append({"text": text, "probability": p, "freq": freqs[text]})
        for g in groups.values():
            if g["sense"] is not None:
                g["label"] = self.senses.get(g["key"], {}).get(str(g["sense"]), {}).get("label", "")
            elif len(g["readings"]) == 1 and g["readings"][0]["text"] == number_reading(surface):
                g["label"] = f"the number {numeral_value(clean(surface))}"      # the numeral read as itself, not as an abbreviation
                g["kind"] = "number"
            g["readings"].sort(key=lambda r: -r["probability"])
            g["text"] = g["readings"][0]["text"]
        return sorted(groups.values(), key=lambda g: -g["probability"])

    # ---- encoding ---------------------------------------------------------------------------------------------------------------

    @torch.inference_mode()
    def _encode_queries(self, rows: list[dict]) -> torch.Tensor:
        encodings = [encode_query(self.tokenizer, r["before"], r["surface"], r["after"], "", "", max_length=MAX_LENGTH, context_words=CONTEXT_WORDS) for r in rows]
        batch = {k: v.to(self.device) for k, v in collate_queries(self.tokenizer, encodings).items()}
        hidden = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], token_type_ids=batch["token_type_ids"]).last_hidden_state
        mask = (batch["span_mask"] if self.pooling == "span" else batch["attention_mask"]).unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return torch.nn.functional.normalize(pooled.float(), dim=-1).cpu()

    @torch.inference_mode()
    def _expansion_vectors(self, texts: list[str]) -> torch.Tensor:
        """Embeddings of the readings: from the index where it has them, else (citation readings) computed now."""
        missing = [t for t in dict.fromkeys(texts) if t not in self.index.position and t not in self._embeddings]
        for i in range(0, len(missing), 64):
            chunk = missing[i:i + 64]
            batch = {k: v.to(self.device) for k, v in collate_candidates(self.tokenizer, [encode_candidate(self.tokenizer, t) for t in chunk]).items()}
            hidden = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = torch.nn.functional.normalize(((hidden * mask).sum(1) / mask.sum(1).clamp(min=1)).float(), dim=-1).cpu()
            for t, v in zip(chunk, pooled):
                self._embeddings[t] = v
        return torch.stack([self.index.embeddings[self.index.position[t]] if t in self.index.position else self._embeddings[t] for t in texts])

    # ---- one token / a text -----------------------------------------------------------------------------------------------------

    def candidates(self, surface: str) -> dict[str, int]:
        """{reading: frequency}: the lexicon's candidates plus citation readings (not for a bare numeral nor a token ending in a geresh)."""
        pattern = {} if (numeral_value(clean(surface)) is not None or surface.endswith("'")) else citation_readings(surface)
        return {**pattern, **self.index.lexicon.candidates(surface)}

    def score(self, rows: list[dict], prior_weight: Optional[float] = None) -> list[dict]:
        """rows: [{surface, before, after}] -> per row {senses: [sense groups, best first], n_readings} (no senses when the lexicon offers nothing for the token)."""
        w = self.prior_weight if prior_weight is None else prior_weight
        results: list[dict] = [{"senses": [], "n_readings": 0} for _ in rows]
        live = [(i, found) for i, r in enumerate(rows) if (found := self.candidates(r["surface"]))]
        if not live:
            return results
        queries = self._encode_queries([rows[i] for i, _ in live])
        for (i, found), q in zip(live, queries):
            texts = list(found)
            logits = self.scale * (self._expansion_vectors(texts) @ q) + w * torch.tensor([prior(found[t]) for t in texts])
            probs = torch.softmax(logits, dim=0).tolist()
            results[i] = {"senses": self._describe(rows[i]["surface"], dict(zip(texts, probs)), found), "n_readings": len(texts)}
        return results

    def find(self, text: str) -> tuple[str, list[dict]]:
        """Clean `text` line by line and detect abbreviation tokens. Returns (cleaned text, tokens with `start` / `end` offsets in it)."""
        lines = [clean(line) for line in text.replace("\r", "").split("\n")]
        tokens, offset = [], 0
        for line in lines:
            for d in detect(line):
                if is_non_abbreviation(d["surface"]):
                    continue                       # letter names / divine names with gershayim are never abbreviations
                d["start"] += offset
                d["end"] = d["start"] + len(d["surface"])
                tokens.append(d)
            offset += len(line) + 1
        return "\n".join(lines), tokens

    def expand(self, text: str, prior_weight: Optional[float] = None) -> dict:
        """{text (cleaned), tokens: [{surface, start, end, senses, n_readings, ...}]}; `senses[0]` is the answer (`senses[0]["text"]` its most likely spelling)."""
        cleaned, tokens = self.find(text)
        rows = [{"surface": t["surface"], "before": cleaned[:t["start"]], "after": cleaned[t["end"]:]} for t in tokens]
        for t, res in zip(tokens, self.score(rows, prior_weight)):
            t.update(res)
        return {"text": cleaned, "tokens": tokens}
