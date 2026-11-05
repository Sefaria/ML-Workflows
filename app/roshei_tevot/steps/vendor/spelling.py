# Vendored from Sefaria/translation-engine experiments/expand_roshei_tevot @ b8160e4f: llm_pass.stem_key and fold_new_senses.fold_b
# (the two string rules behind full_run.matches), copied without their module dependencies. Re-sync rather than edit.
"""Do two expansions mean the same thing, differing only in spelling?

  same_key  per word: one leading prefix letter dropped (when the word is longer than 3), final forms unified, non-initial י/ו dropped
  fold_b    same number of words, each pair equal after stripping up to 3 attached prefixes (ו ה ב כ ל מ ש ד) and folding י/ו,
            or stems sharing 3+ letters that differ by at most 2 (עובדיה ברטנורא = עובדיה מברטנורא)

Pairs these rules keep apart on purpose: רבי יוסה / רבי יוסי and similar final-letter variants (a general fold would merge רבא / רבי).
"""
import re

from .hebrew import PREFIX_LETTERS, medial

FOLD_PREFIXES = "והבכלמשד"


def stem_key(expansion: str) -> tuple[str, ...]:
    words = [medial(w) for w in re.sub(r"[^\w\s]", "", expansion).split()]
    words = [w[1:] if len(w) > 3 and w[0] in PREFIX_LETTERS else w for w in words]
    return tuple(w[:1] + re.sub("[יו]", "", w[1:]) for w in words)


def _fold(word: str) -> str:
    return word[:1] + re.sub("[יו]", "", word[1:])


def _strips(word: str) -> set[str]:
    out, current = {_fold(word)}, word
    for _ in range(3):
        if len(current) > 3 and current[0] in FOLD_PREFIXES:
            current = current[1:]
            out.add(_fold(current))
        else:
            break
    return out


def _close(x: str, y: str) -> bool:
    if x == y:
        return True
    n = 0
    while n < min(len(x), len(y)) and x[n] == y[n]:
        n += 1
    return n >= 3 and len(x) - n <= 2 and len(y) - n <= 2


def fold_b(a: str, b: str) -> bool:
    words_a = [medial(w) for w in re.sub(r"[^\w\s]", "", a).split()]
    words_b = [medial(w) for w in re.sub(r"[^\w\s]", "", b).split()]
    if len(words_a) != len(words_b) or not words_a:
        return False
    for x, y in zip(words_a, words_b):
        sx, sy = _strips(x), _strips(y)
        if not (sx & sy or any(_close(p, q) for p in sx for q in sy)):
            return False
    return True


def same_expansion(a: str, b: str) -> bool:
    return a == b or stem_key(a) == stem_key(b) or fold_b(a, b)
