"""The lexicon (which expansions a token can have) and the expansion index (the lexicon plus one embedding per expansion).

Candidates for a token are the expansions the lexicon lists for the token as written or for any prefix-stripped reading of it, minus
the ones whose letters do not spell the token (the vendored checker). Training mines its hard negatives from the same lookup, so it
practises exactly the choice it makes at inference. About 57k expansions at 768 dimensions is under 100 MB in fp16, so the index lives in RAM.
"""
import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .vendor import token_fits
from .vendor.detect import parses
from .vendor.hebrew import clean


@lru_cache(maxsize=1_000_000)
def fits_cached(surface: str, expansion: str) -> bool:
    return token_fits(surface, expansion)


class Lexicon:
    def __init__(self, by_surface: dict[str, dict[str, int]], by_bare: dict[str, dict[str, int]]):
        self.by_surface = by_surface
        self.by_bare = by_bare
        self._cache: dict[str, dict[str, int]] = {}

    @classmethod
    def load(cls, path: str) -> "Lexicon":
        """Read `surface_index.jsonl`: one line per surface with its candidate expansions and their library frequency."""
        by_surface: dict[str, dict[str, int]] = {}
        by_bare: dict[str, dict[str, int]] = defaultdict(dict)
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                counts = by_surface.setdefault(row["surface"], {})
                for c in row["candidates"]:
                    counts[c["expansion"]] = counts.get(c["expansion"], 0) + c["freq"]
                    bare = by_bare[c["bare"]]
                    bare[c["expansion"]] = bare.get(c["expansion"], 0) + c["freq"]
        return cls(by_surface, dict(by_bare))

    def expansions(self) -> list[str]:
        seen: dict[str, None] = {}
        for table in (self.by_surface, self.by_bare):
            for counts in table.values():
                seen.update(dict.fromkeys(counts))
        return list(seen)

    def candidates(self, surface: str) -> dict[str, int]:
        """{expansion: lexicon frequency} the lexicon lists for this token and whose letters fit it. Cached per surface."""
        surface = clean(surface)
        if surface not in self._cache:
            pool: dict[str, int] = dict(self.by_surface.get(surface, {}))
            for _prefix, bare in parses(surface):
                for text, freq in self.by_bare.get(bare, {}).items():
                    pool[text] = max(pool.get(text, 0), freq)
            self._cache[surface] = {text: freq for text, freq in pool.items() if fits_cached(surface, text)}
        return self._cache[surface]

    def scan(self, surface: str) -> dict[str, int]:
        """Every known expansion whose letters fit the token, for tokens the lexicon does not list. Slow: one checker call per expansion."""
        surface = clean(surface)
        return {text: 0 for text in self.expansions() if fits_cached(surface, text)}


def prior(freq: int) -> float:
    return math.log1p(freq)


class ExpansionIndex:
    def __init__(self, lexicon: Lexicon, expansions: list[str], embeddings: torch.Tensor):
        self.lexicon = lexicon
        self.expansions = expansions
        self.embeddings = embeddings
        self.position = {text: i for i, text in enumerate(expansions)}

    @classmethod
    def build(cls, lexicon: Lexicon, model, tokenizer, device: str) -> "ExpansionIndex":
        from .data import encode_texts

        texts = lexicon.expansions()
        return cls(lexicon, texts, encode_texts(model, tokenizer, texts, device))

    def save(self, directory: str) -> None:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "embeddings.npy", self.embeddings.to(torch.float16).numpy())
        (out / "lexicon.json").write_text(json.dumps(
            {"expansions": self.expansions, "by_surface": self.lexicon.by_surface, "by_bare": self.lexicon.by_bare}, ensure_ascii=False))

    @classmethod
    def load(cls, directory: str) -> "ExpansionIndex":
        data = json.loads((Path(directory) / "lexicon.json").read_text())
        embeddings = torch.from_numpy(np.load(Path(directory) / "embeddings.npy")).float()
        return cls(Lexicon(data["by_surface"], data["by_bare"]), data["expansions"], embeddings)

    def rank(self, surface: str, query: torch.Tensor, scale: float = 1.0, prior_weight: float = 0.0,
             scan_unknown: bool = False) -> list[tuple[str, float, int]]:
        """[(expansion, score, frequency)] best first; score = scale * cosine + prior_weight * log(1 + frequency). Empty when nothing fits."""
        found = self.lexicon.candidates(surface) or (self.lexicon.scan(surface) if scan_unknown else {})
        if not found:
            return []
        texts = list(found)
        scores = scale * (self.embeddings[[self.position[t] for t in texts]] @ query)
        scores = scores + prior_weight * torch.tensor([prior(found[t]) for t in texts])
        order = torch.argsort(scores, descending=True).tolist()
        return [(texts[i], float(scores[i]), found[texts[i]]) for i in order]


def decide(ranked: list[tuple[str, float, int]], min_score: float, min_margin: float) -> Optional[str]:
    """Soft discard: answer with the top expansion unless it is too weak or too close to the runner-up; None means abstain."""
    if not ranked:
        return None
    top = ranked[0][1]
    margin = top - ranked[1][1] if len(ranked) > 1 else float("inf")
    return ranked[0][0] if top >= min_score and margin >= min_margin else None
