# Vendored from Sefaria/translation-engine experiments/expand_roshei_tevot/hebrew.py @ b8160e4f; only imports were made relative. Re-sync rather than edit.
"""Hebrew text helpers shared by the detector, the checker and the lexicon: normalisation, gematria, final forms."""
import html, re

TAG = re.compile(r"<[^>]+>")
NIKUD = re.compile("[\u0591-\u05bd\u05bf-\u05c7]")   # cantillation + nikud, but not the maqaf (U+05BE): it separates words
FINALS = str.maketrans("ךםןףץ", "כמנפצ")
PREFIX_LETTERS = "ובהכלמש"

def normalize_punct(s):
    """Unify gershayim variants to `"` and geresh variants to `'`; a maqaf becomes a space (דבר־ה' is two words)."""
    s = s.replace("\u05be", " ").replace("״", '"').replace("“", '"').replace("”", '"').replace("''", '"')
    return s.replace("׳", "'").replace("’", "'").replace("‘", "'")

BIDI = re.compile("[\u200e\u200f\u200d\u202a-\u202e]")

def clean(s):
    """Strip tags, entities, nikud and bidi marks; unify punctuation; collapse whitespace."""
    s = BIDI.sub("", html.unescape(TAG.sub(" ", s))).replace('\\"', '"')
    return re.sub(r"\s+", " ", normalize_punct(NIKUD.sub("", s))).strip()

def medial(s):
    return s.translate(FINALS)

# ---- gematria ----
VALUES = {c: v for c, v in zip("אבגדהוזחטיכלמנסעפצקרשת",
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400])}
VALUES.update({"ך": 20, "ם": 40, "ן": 50, "ף": 80, "ץ": 90})

def gematria_value(letters):
    """Numeric value of a Hebrew-numeral spelling, or None if the letters are not a valid numeral.
    Valid: values non-increasing; at most one tens letter and one units letter; hundreds repeat only as ת (תת=800);
    15/16 are written ט"ו / ט"ז, never י"ה / י"ו (which are divine-name spellings)."""
    if not letters or any(c not in VALUES for c in letters):
        return None
    if letters in ("טו", "טז"):
        return VALUES[letters[0]] + VALUES[letters[1]]
    vals = [VALUES[c] for c in letters]
    if vals != sorted(vals, reverse=True):
        return None
    if sum(1 for v in vals if 10 <= v < 100) > 1 or sum(1 for v in vals if v < 10) > 1:
        return None
    other_hundreds = [c for c in letters if VALUES[c] >= 100 and c != "ת"]
    if len(other_hundreds) != len(set(other_hundreds)):
        return None
    if letters.endswith(("יה", "יו")):
        return None
    return sum(vals)


def numeral_value(abbr):
    """Numeric reading of a candidate token, or None. A geresh marks a numeral only after a single letter (ה'); two or more
    letters need gershayim before the last one (י"ד), so a multi-letter token with only a trailing geresh is not a numeral."""
    letters = abbr.replace('"', "").replace("'", "")
    if len(letters) > 1 and '"' not in abbr:
        return None
    return gematria_value(letters)
