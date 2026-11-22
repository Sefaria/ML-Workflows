# Vendored from Sefaria/translation-engine experiments/expand_roshei_tevot/numwords.py @ 0af8ac5; only imports were made relative. Re-sync rather than edit.
"""Hebrew number words <-> value <-> gematria numeral, for the `number` reading of a numeral token.

    words_value("ארבעה עשר") == 14        parse spelled-out Hebrew numbers, 1-9999, either gender, classical or modern spelling (שלשה / שלושה)
    value_words(14)          == "ארבעה עשר"   the canonical (masculine) spelling used as the number reading in training rows and candidates
    value_numeral(14)        == 'י"ד'     the gematria spelling (1-999; None above)

A numeral token such as י"ד has two kinds of reading: abbreviations (יורה דעה) and the number itself. The number reading is the spelled-out words, so the
encoder sees ordinary Hebrew and a Dicta answer ("ארבעה עשר") compares by value, whatever the gender or spelling.
"""
import re

from .hebrew import VALUES, medial

UNITS_M = ["", "אחד", "שנים", "שלשה", "ארבעה", "חמשה", "ששה", "שבעה", "שמונה", "תשעה"]
UNITS_F = ["", "אחת", "שתים", "שלש", "ארבע", "חמש", "שש", "שבע", "שמונה", "תשע"]
TENS = ["", "עשרה", "עשרים", "שלשים", "ארבעים", "חמשים", "ששים", "שבעים", "שמונים", "תשעים"]
TEENS_M = {10: "עשרה", 11: "אחד עשר", 12: "שנים עשר", 13: "שלשה עשר", 14: "ארבעה עשר", 15: "חמשה עשר", 16: "ששה עשר", 17: "שבעה עשר", 18: "שמונה עשר", 19: "תשעה עשר"}


def _norm(word):
    """The word's Hebrew letters with final forms made medial. Spelling variants are listed explicitly (no folding of ו / י: שמו is not ששים)."""
    return medial(re.sub(r"[^א-ת]", "", word))


_WORD = {}          # normalised spelling -> (kind, value)


def _add(words, kind, value):
    for w in words.split():
        _WORD.setdefault(_norm(w), (kind, value))


_add("אחד אחת", "unit", 1)
_add("שנים שניים שתים שתיים שתי שני", "unit", 2)
_add("שלשה שלושה שלש שלוש", "unit", 3); _add("ארבעה ארבע", "unit", 4); _add("חמשה חמישה חמש", "unit", 5); _add("ששה שישה שש", "unit", 6)
_add("שבעה שבע", "unit", 7); _add("שמונה שמנה", "unit", 8); _add("תשעה תשע", "unit", 9)
_add("שלשת שלושת", "unit_c", 3); _add("ארבעת", "unit_c", 4); _add("חמשת חמישת", "unit_c", 5); _add("ששת שישת", "unit_c", 6)
_add("שבעת", "unit_c", 7); _add("שמונת שמנת", "unit_c", 8); _add("תשעת", "unit_c", 9)
_add("עשר עשרה", "ten", 10)
_add("עשרים", "tens", 20); _add("שלשים שלושים", "tens", 30); _add("ארבעים", "tens", 40); _add("חמשים חמישים", "tens", 50)
_add("ששים שישים", "tens", 60); _add("שבעים", "tens", 70); _add("שמונים שמנים", "tens", 80); _add("תשעים", "tens", 90)
_add("מאה", "h1", 100); _add("מאתים מאתיים", "h2", 200); _add("מאות", "hw", 100)
_add("אלף", "k1", 1000); _add("אלפיים", "k2", 2000); _add("אלפים", "kw", 1000)


def is_number_word(word):
    """The word, exactly as spelled, is one of the number words (no attached ו)."""
    return _norm(word) in _WORD


def _tokens(text):
    out = []
    for w in (w for w in re.split(r"[\s\-־]+", text.strip()) if w):
        n, vav = _norm(w), False
        if n not in _WORD and len(n) > 2 and n[0] == "ו" and n[1:] in _WORD:
            n, vav = n[1:], True                        # ושלשה: the conjunction on a later word
        if n not in _WORD:
            return None
        out.append((*_WORD[n], vav))
    return out or None


def words_value(text):
    """The value of a spelled-out Hebrew number in 1-9999, or None if the text is not exactly one number.
    Grammar: [thousands] [hundreds] [below 100], magnitudes descending: אלף / אלפיים / <3-9> אלפים; מאה / מאתיים / <3-9> מאות;
    below 100: <unit> | עשר(ה) | <unit> עשר(ה) (11-19) | <tens> [<unit>] | <unit> <tens> (ארבעה ועשרים).
    A number of two or more components (מאה ועשרים, עשרים ושנים, אלף ושלש מאות) carries the conjunction ו on its last component, as spelled: that is what keeps
    the noun שנים (years) after a number from being read as 2 (עשרים שנים is twenty years, not 22). Teens and <3-9> מאות / אלפים are one component."""
    toks = _tokens(text)
    if toks is None or len(toks) > 9:
        return None
    i, total = 0, 0
    starts = []                                         # index of the first token of each additive component

    def peek(*kinds):
        return i < len(toks) and toks[i][0] in kinds

    if peek("k1", "k2"):
        starts.append(i); total += toks[i][1]; i += 1
    elif i + 1 < len(toks) and toks[i][0] in ("unit", "unit_c") and 3 <= toks[i][1] <= 9 and toks[i + 1][0] == "kw":
        starts.append(i); total += toks[i][1] * 1000; i += 2
    if peek("h1", "h2"):
        starts.append(i); total += toks[i][1]; i += 1
    elif i + 1 < len(toks) and toks[i][0] == "unit" and 3 <= toks[i][1] <= 9 and toks[i + 1][0] == "hw":
        starts.append(i); total += toks[i][1] * 100; i += 2
    if peek("unit"):
        starts.append(i); unit_idx, unit = i, toks[i][1]; i += 1
        if peek("ten"):
            total += 10 + unit; i += 1
        elif peek("tens"):
            if len(starts) > 1 and not toks[unit_idx][2]:
                return None                                              # after hundreds the unit-first pair is ושנים ועשרים, not שנים ועשרים
            starts.append(i); total += unit + toks[i][1]; i += 1          # ארבעה ועשרים
        else:
            total += unit
    elif peek("ten"):
        starts.append(i); total += 10; i += 1
    elif peek("tens"):
        starts.append(i); total += toks[i][1]; i += 1
        if peek("unit"):
            starts.append(i); total += toks[i][1]; i += 1
    if i != len(toks) or not 1 <= total <= 9999:
        return None
    if len(starts) >= 2 and not toks[starts[-1]][2]:
        return None
    return total


def _below_100(n):
    if n < 10:
        return UNITS_M[n]
    if n < 20:
        return TEENS_M[n]
    tens, units = divmod(n, 10)
    return TENS[tens] if not units else f"{TENS[tens]} ו{UNITS_M[units]}"


def value_words(n):
    """Canonical spelling of 1-9999: masculine units, "ו" before the last word (מאה ועשרים, שלש מאות וארבעים וחמשה)."""
    if not 1 <= n <= 9999:
        raise ValueError(n)
    parts = []
    thousands, rest = divmod(n, 1000)
    if thousands:
        parts.append("אלף" if thousands == 1 else "אלפיים" if thousands == 2 else f"{UNITS_F[thousands]} אלפים")
    hundreds, below = divmod(rest, 100)
    if hundreds:
        parts.append("מאה" if hundreds == 1 else "מאתים" if hundreds == 2 else f"{UNITS_F[hundreds]} מאות")
    if below:
        parts.append(_below_100(below))
    if len(parts) > 1 and not parts[-1].startswith("ו"):
        parts[-1] = "ו" + parts[-1]
    return " ".join(parts)


def value_numeral(n):
    """Gematria spelling with the geresh / gershayim marks (ה', י"ד, ט"ו, קכ"ג, קט"ו, תתקפ"ו); None outside 1-999."""
    if not 1 <= n <= 999:
        return None
    letters = []
    for v in (400, 300, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1):
        while n >= v:
            letters.append(next(c for c, x in VALUES.items() if x == v and c not in "ךםןףץ")); n -= v
    s = "".join(letters)
    s = s.replace("יה", "טו").replace("יו", "טז")
    return s + "'" if len(s) == 1 else s[:-1] + '"' + s[-1]
