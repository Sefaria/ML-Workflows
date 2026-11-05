"""The expansion database: every known expansion embedded once, plus the lexicon's map from abbreviations to expansions.

Lookup for a token is: the expansions the lexicon lists for the token as written or for any prefix-stripped reading of it, minus
the ones whose letters do not spell the token (the vendored checker), ranked by cosine similarity to the context embedding.
About 57k expansions at 768 dimensions is under 100 MB in fp16, so the whole index lives in RAM.
"""
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .data import encode_texts
from .vendor import token_fits
from .vendor.detect import parses
from .vendor.hebrew import clean


@lru_cache(maxsize=500_000)
def fits_cached(surface: str, expansion: str) -> bool:
    return token_fits(surface, expansion)


class ExpansionIndex:
    def __init__(self, expansions: list[str], embeddings: torch.Tensor, by_surface: dict[str, dict[str, int]], by_bare: dict[str, dict[str, int]]):
        self.expansions = expansions
        self.embeddings = embeddings
        self.by_surface = by_surface
        self.by_bare = by_bare
        self.position = {text: i for i, text in enumerate(expansions)}

    # ---- build / persist ----
    @classmethod
    def build(cls, lexicon_path: str, model, tokenizer, device: str) -> "ExpansionIndex":
        by_surface: dict[str, dict[str, int]] = {}
        by_bare: dict[str, dict[str, int]] = defaultdict(dict)
        expansions: dict[str, int] = {}
        with open(lexicon_path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                counts = by_surface.setdefault(row["surface"], {})
                for c in row["candidates"]:
                    expansions.setdefault(c["expansion"], len(expansions))
                    counts[c["expansion"]] = counts.get(c["expansion"], 0) + c["freq"]
                    bare = by_bare[c["bare"]]
                    bare[c["expansion"]] = bare.get(c["expansion"], 0) + c["freq"]
        texts = list(expansions)
        return cls(texts, encode_texts(model, tokenizer, texts, device), by_surface, dict(by_bare))

    def save(self, directory: str) -> None:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "embeddings.npy", self.embeddings.to(torch.float16).numpy())
        (out / "lexicon.json").write_text(json.dumps({"expansions": self.expansions, "by_surface": self.by_surface, "by_bare": self.by_bare}, ensure_ascii=False))

    @classmethod
    def load(cls, directory: str) -> "ExpansionIndex":
        data = json.loads((Path(directory) / "lexicon.json").read_text())
        embeddings = torch.from_numpy(np.load(Path(directory) / "embeddings.npy")).float()
        return cls(data["expansions"], embeddings, data["by_surface"], data["by_bare"])

    # ---- lookup ----
    def candidates(self, surface: str, scan_unknown: bool = False) -> dict[str, int]:
        """{expansion: lexicon frequency} that the lexicon lists for this token and whose letters fit it."""
        surface = clean(surface)
        pool: dict[str, int] = dict(self.by_surface.get(surface, {}))
        for _prefix, bare in parses(surface):
            for text, freq in self.by_bare.get(bare, {}).items():
                pool[text] = max(pool.get(text, 0), freq)
        kept = {text: freq for text, freq in pool.items() if fits_cached(surface, text)}
        if not kept and scan_unknown:
            kept = {text: 0 for text in self.expansions if fits_cached(surface, text)}
        return kept

    def rank(self, surface: str, query: torch.Tensor, scan_unknown: bool = False) -> list[tuple[str, float, int]]:
        """[(expansion, cosine, frequency)] best first. Empty when nothing in the database fits the token."""
        found = self.candidates(surface, scan_unknown)
        if not found:
            return []
        texts = list(found)
        scores = self.embeddings[[self.position[t] for t in texts]] @ query
        order = torch.argsort(scores, descending=True).tolist()
        return [(texts[i], float(scores[i]), found[texts[i]]) for i in order]


def decide(ranked: list[tuple[str, float, int]], min_score: float, min_margin: float) -> Optional[str]:
    """Soft discard: answer with the top expansion unless it is too weak or too close to the runner-up; None means abstain."""
    if not ranked:
        return None
    top = ranked[0][1]
    margin = top - ranked[1][1] if len(ranked) > 1 else float("inf")
    return ranked[0][0] if top >= min_score and margin >= min_margin else None
