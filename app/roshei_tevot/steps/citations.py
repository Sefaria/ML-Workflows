"""Citation readings (`סקי"א` = סעיף קטן י"א): HEAD + NUMERAL patterns from expand_roshei_tevot/citation_patterns.json, vendored in rt/vendor/.
The lexicon stores these as patterns, not as one entry per number, so a candidate index built from it never lists them; this expands them per token."""
import collections
import json
import re
from pathlib import Path

from .vendor.hebrew import gematria_value

MARK = re.compile(r"[\"'״׳]")
_index = None


def _strip(s: str) -> str:
    return MARK.sub("", s)


def _fmt_numeral(letters: str) -> str:
    return letters + "'" if len(letters) == 1 else letters[:-1] + '"' + letters[-1]


def expand(surface: str) -> dict[str, int]:
    """{reading text: pattern frequency per number} for every pattern whose head plus a numeral tail spell the token."""
    global _index
    if _index is None:
        _index = collections.defaultdict(list)
        for r in json.loads((Path(__file__).parent / "vendor" / "citation_patterns.json").read_text(encoding="utf-8")):
            _index[r["head"]].append(r)
    s, out = _strip(surface), {}
    for i in range(len(s)):
        head, tail = s[:i], s[i:]
        if gematria_value(tail) is None:
            continue
        for r in _index.get(head, ()):
            out[f"{r['prefix']}{r['unit']} {_fmt_numeral(tail)}"] = max(1, round(r["freq"] / max(1, r["entries"])))
    return out
