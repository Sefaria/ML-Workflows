"""The lexicon (which expansions a token can have) and the expansion index (the lexicon plus one embedding per expansion).

Candidates for a token are the expansions the lexicon lists for the token as written or for any prefix-stripped reading of it, minus
the ones whose letters do not spell the token (the vendored checker). Training mines its hard negatives from the same lookup, so it
practises exactly the choice it makes at inference. About 57k expansions at 768 dimensions is under 100 MB in fp16, so the index lives in RAM.
"""
import bisect
import json
import math
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .vendor import token_fits
from .vendor.detect import parses
from .vendor.hebrew import clean, numeral_value
from .vendor.numwords import value_numeral, value_words


def number_reading(surface: str) -> Optional[str]:
    """The number a token spells, as words, when the token is a numeral of two or more letters exactly as written (י"ד -> ארבעה עשר, קט"ו -> מאה וחמשה עשר), else None.
    A numeral is always a possible reading of itself, whether or not the lexicon lists an abbreviation for the same letters. One-letter numerals with a geresh (ג')
    are left to the lexicon, which reads them as number words (א' = אחד); a prefix before a numeral (בכ"ד) is stripped by the caller (Lexicon.candidates)."""
    value = numeral_value(surface)
    if value is None or not 11 <= value <= 999 or len(surface.replace('"', "").replace("'", "")) < 2:
        return None
    return value_words(value)


GERESH_LETTER = re.compile(r"[אבגדהוזחטיכלמנסעפצקרשת]'")
SK_NUMERAL = re.compile(r'^סק("?[א-ת]"?[א-ת]*)$')


def geresh_number_reading(bare: str) -> Optional[str]:
    """The number a one-letter numeral with a geresh spells (ג' -> שלשה, מ' -> ארבעים, ק' -> מאה): a count when a counted noun follows (בג' ימים), a letter otherwise.
    The model decides; the number is always one of the candidates, with or without an attached prefix (Lexicon.candidates strips it)."""
    if not GERESH_LETTER.fullmatch(bare):
        return None
    value = numeral_value(bare)
    return value_words(value) if value else None


def sk_reading(bare: str) -> Optional[str]:
    """סק"ח -> סעיף קטן ח', סקי"ג -> סעיף קטן י"ג: the commentary's small paragraph, which the letters ק + numeral also spell as a larger numeral or as סעיף ק"ח."""
    m = SK_NUMERAL.match(bare)
    if not m:
        return None
    letters = m.group(1).replace('"', "")
    tail = letters + "'" if len(letters) == 1 else letters[:-1] + '"' + letters[-1]
    return f"סעיף קטן {tail}" if numeral_value(tail) else None


SK_TEXTS = [f"סעיף קטן {value_numeral(n)}" for n in range(1, 1000)]
NUMBER_WORDS = [value_words(n) for n in range(1, 1000)] + SK_TEXTS    # every number reading, so the expansion index holds an embedding for each
COMPLETION_MIN_STEM, COMPLETION_TOP = 3, 8


TRUSTED_FREQ = 3       # a reading the corpus labels at least this often is a candidate without passing the letter checker (set to 10**9 to turn this off)


@lru_cache(maxsize=1_000_000)
def fits_cached(surface: str, expansion: str) -> bool:
    return token_fits(surface, expansion)


class Lexicon:
    def __init__(self, by_surface: dict[str, dict[str, int]], by_bare: dict[str, dict[str, int]], vocab: Optional[dict[str, int]] = None):
        self.by_surface = by_surface
        self.by_bare = by_bare
        self.vocab = vocab or {}                       # word -> library frequency, the words a truncated token (לכתחי') can complete to
        self._words = sorted(self.vocab)
        self._cache: dict[str, dict[str, int]] = {}

    @classmethod
    def load(cls, path: str, vocab_path: Optional[str] = None) -> "Lexicon":
        """Read `surface_index.jsonl`: one line per surface with its candidate expansions and their library frequency; `vocab_path` is `vocab.jsonl` ({"word", "freq"} per line)."""
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
        vocab: dict[str, int] = {}
        if vocab_path:
            with open(vocab_path, encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    vocab[row["word"]] = row["freq"]
        return cls(by_surface, dict(by_bare), vocab)

    def completions(self, stem: str) -> dict[str, int]:
        """The most frequent vocabulary words that start with the stem and are longer than it (לכתחי -> לכתחילה's bare form כתחילה, after the prefix is stripped)."""
        if len(stem) < COMPLETION_MIN_STEM or not self._words:
            return {}
        i = bisect.bisect_left(self._words, stem)
        out = []
        while i < len(self._words) and self._words[i].startswith(stem):
            if len(self._words[i]) > len(stem):
                out.append(self._words[i])
            i += 1
        out.sort(key=lambda w: -self.vocab[w])
        return {w: self.vocab[w] for w in out[:COMPLETION_TOP]}

    def expansions(self) -> list[str]:
        seen: dict[str, None] = {}
        for table in (self.by_surface, self.by_bare):
            for counts in table.values():
                seen.update(dict.fromkeys(counts))
        seen.update(dict.fromkeys(NUMBER_WORDS))
        seen.update(dict.fromkeys(self.vocab))
        return list(seen)

    def candidates(self, surface: str) -> dict[str, int]:
        """{expansion: lexicon frequency} the lexicon lists for this token and whose letters fit it, or that the corpus attests at least TRUSTED_FREQ times (the checker
        caps a word at 3 letters, so it rejects real abbreviations that spell whole words: קיימ"ל, יעוי"ש, פיקו"נ, תום'; 530 such readings have freq >= 3). Cached per surface."""
        surface = clean(surface)
        if surface not in self._cache:
            pool: dict[str, int] = dict(self.by_surface.get(surface, {}))
            for _prefix, bare in parses(surface):
                for text, freq in self.by_bare.get(bare, {}).items():
                    pool[text] = max(pool.get(text, 0), freq)
            found = {text: freq for text, freq in pool.items() if freq >= TRUSTED_FREQ or fits_cached(surface, text)}
            for _prefix, bare in parses(surface):           # a numeral (also behind a prefix: לי"ב, וקי"ד) can always be the number; the model decides between it and the abbreviation readings
                for reading in (number_reading(bare), geresh_number_reading(bare), sk_reading(bare)):
                    if reading:
                        found.setdefault(reading, 0)
                if bare.endswith("'") and '"' not in bare:        # a truncated word (לכתחי'): the library's words that start with it
                    for word, freq in self.completions(bare[:-1]).items():
                        found.setdefault(word, freq)
            self._cache[surface] = found
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
            {"expansions": self.expansions, "by_surface": self.lexicon.by_surface, "by_bare": self.lexicon.by_bare, "vocab": self.lexicon.vocab}, ensure_ascii=False))

    @classmethod
    def load(cls, directory: str) -> "ExpansionIndex":
        data = json.loads((Path(directory) / "lexicon.json").read_text())
        embeddings = torch.from_numpy(np.load(Path(directory) / "embeddings.npy")).float()
        return cls(Lexicon(data["by_surface"], data["by_bare"], data.get("vocab")), data["expansions"], embeddings)

    def rank(self, surface: str, query: torch.Tensor, scale: float = 1.0, prior_weight: float = 0.0,
             scan_unknown: bool = False) -> list[tuple[str, float, int]]:
        """[(expansion, score, frequency)] best first; score = scale * cosine + prior_weight * log(1 + frequency). Empty when nothing fits."""
        found = self.lexicon.candidates(surface) or (self.lexicon.scan(surface) if scan_unknown else {})
        found = {text: freq for text, freq in found.items() if text in self.position}      # an index built before number readings existed has no embedding for them
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
