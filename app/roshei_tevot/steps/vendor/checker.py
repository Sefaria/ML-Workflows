# Vendored from Sefaria/translation-engine experiments/expand_roshei_tevot/checker.py @ b8160e4f; only imports were made relative. Re-sync rather than edit.
"""Hand-written alignment checker: does this expansion fit these abbreviation letters?

    fits(abbr, expansion) -> (bool, alignment)

Rules (PLAN step 2):
  * every abbreviation letter comes from a word of the expansion, in order, and every word contributes
    1..MAX_RUN consecutive leading letters of its stem (הקב"ה: הקדוש -> ה,ק; חז"ל: one letter per word);
  * a word may carry omitted prefix letters (ו ה ב כ ל מ ש, up to 3), so the stem starts after them; the
    abbreviation may also spell the prefix letter itself (stem p = 0 matches a leading ו/ה/...);
  * the abbreviation itself may open with up to MAX_ABBR_PREFIX attached prefix letters that belong to no expansion
    word (הרא"ש = (ה)רבי אשר) -- tried only after a strict fit fails; Aramaic ד counts as a stem prefix (דקהילה);
  * final forms are medial on both sides;
  * a trailing geresh on the abbreviation allows truncation: its letters are a prefix of one word (וכו' = וכולי);
  * a small whitelist covers lexicalised exceptions that break the rules;
  * LENIENT_MATRES (emot kriah), tried only after the strict fit fails and only in two places: inside a run, an expansion word may
    carry ו/י between two letters the abbreviation takes (בתס"ר = בתוספת ראשונים); and on a truncated abbreviation (trailing geresh)
    its own non-initial ו/י are ignored against the matres-free word (עירוכי' = ערכין).
"""
import re
from functools import lru_cache
from .hebrew import NIKUD, PREFIX_LETTERS, medial, normalize_punct

MAX_RUN = 3
MAX_RUN_LAST = 5            # the last word may be truncated harder: ביהכנ"ס = בית הכנסת, אדמוה"ז = אדמור הזקן
MAX_PREFIX = 3
MAX_ABBR_PREFIX = 0         # 1 lifts kizur acceptance ~3 pts but doubles near-miss false accepts; the detector's prefix parses cover it
SINGLE_WORD_RUNS = True     # may a one-word expansion supply several letters (א"ו = אורים)? Off cuts near-miss false accepts
STEM_PREFIXES = PREFIX_LETTERS + "ד"      # ד is the Aramaic "of" prefix (אבדק"ק = אב בית דין דקהילה קדושה)
IRREGULAR = {}            # {abbr: {expansion, ...}} from irregular.tsv (make_irregular.py); disable with USE_IRREGULAR = False
USE_IRREGULAR = True
LENIENT_MATRES = True       # fallback for emot kriah, see the module docstring

def norm_letters(s):
    s = normalize_punct(NIKUD.sub("", s))
    return medial(re.sub(r"[^א-ת]", "", s))

def words(expansion):
    expansion = re.sub(r"\([^)]*\)", " ", normalize_punct(NIKUD.sub("", expansion)))
    out = [medial(re.sub(r"[^א-ת]", "", w)) for w in re.split(r"[\s\-־/]+", expansion)]
    return [w for w in out if w]

def _stems(word):
    """Candidate stem start offsets: 0 and every position after 1..MAX_PREFIX prefix letters."""
    yield 0
    for p in range(1, min(MAX_PREFIX, len(word) - 1) + 1):
        if all(c in STEM_PREFIXES for c in word[:p]):
            yield p
        else:
            break

def _run_end(w, p, seg, lenient):
    """Index in w just past a run matching seg from p (None if no match); lenient skips ו/י between matched letters."""
    j = p
    for k, ch in enumerate(seg):
        if k and lenient:
            while j < len(w) and w[j] != ch and w[j] in "וי":
                j += 1
        if j >= len(w) or w[j] != ch:
            return None
        j += 1
    return j

def _align(letters, ws, lenient=False):
    """DFS: returns an alignment [(word, prefix_len, run_len)] or None. Every word must be used."""
    @lru_cache(maxsize=None)
    def go(i, j):
        if j == len(ws):
            return () if i == len(letters) else None
        if i == len(letters):
            return None
        w = ws[j]
        for p in _stems(w):
            cap = MAX_RUN_LAST if j == len(ws) - 1 else MAX_RUN
            for run in range(1, (cap if SINGLE_WORD_RUNS or len(ws) > 1 else 1) + 1):
                seg = letters[i:i + run]
                if len(seg) < run or _run_end(w, p, seg, lenient) is None:
                    break
                rest = go(i + run, j + 1)
                if rest is not None:
                    return ((w, p, run),) + rest
        return None
    return go(0, 0)

def _fits_letters(letters, ws, trunc):
    al = _align(letters, tuple(ws))
    if al is not None:
        return list(al)
    if trunc and len(ws) == 1:                       # truncation: letters are a prefix of the single word
        for p in _stems(ws[0]):
            if ws[0][p:].startswith(letters):
                return [(ws[0], p, len(letters))]
    return None

def fits(abbr, expansion):
    """Return (ok, alignment). alignment is a list of (word, omitted_prefix_len, letters_taken), or a note string."""
    letters = norm_letters(abbr)
    ws = words(expansion)
    if not letters or not ws:
        return False, None
    if USE_IRREGULAR and expansion in IRREGULAR.get(abbr, ()):
        return True, "irregular"
    trunc = normalize_punct(abbr).endswith("'")
    al = _fits_letters(letters, ws, trunc)
    for n in range(1, MAX_ABBR_PREFIX + 1):          # attached prefix letters on the abbreviation (הרא"ש)
        if al is not None:
            break
        if len(letters) > n + 1 and all(c in PREFIX_LETTERS for c in letters[:n]):
            al = _fits_letters(letters[n:], ws, trunc)
            if al is not None:
                al = [("<abbr prefix>", 0, n)] + al
    if al is None and LENIENT_MATRES:
        al = _fits_lenient(letters, ws, trunc)
    return (True, al) if al is not None else (False, None)

REF_WORDS = {medial(w) for w in ("דף", "עמוד", "פרק", "סימן", "סעיף")}      # words a daf/chapter reference spells out that its abbreviation leaves to one letter

def _reference_fits(surface, expansion):
    """Daf and chapter references: the token's letters are the numeral letters of the expansion plus the side letter, with an optional
    leading letter standing for a reference word (דטו"א = דף טו עמוד א, יז"א = יז עמוד א). Needs at least one reference word in the expansion."""
    letters, ws = norm_letters(surface), words(expansion)
    ref = [w for w in ws if w in REF_WORDS]
    nums = "".join(w for w in ws if w not in REF_WORDS)
    if not ref or not nums or any(len(w) > 4 for w in ws if w not in REF_WORDS):
        return False
    return letters == nums or (letters[1:] == nums and letters[0] in {w[0] for w in ref} | set(STEM_PREFIXES))

def token_fits(surface, expansion):
    """Does `expansion` read the token as written in the text (attached prefix letters, quote marks and all)? The one letter gate:
    full_run.letters_ok, full_run.gated and lexicon.py's checker_ok call it. Three ways in:
      1. the plain checker on any prefix parse of the token (prefix letters that no expansion word carries: הרא"ש = (ה)רבי אשר);
      2. prefix-consistent: the token opens with prefix letters (ו ה ב כ ל מ ש ד, up to 3) and the expansion's first word opens with the same letters,
         so both are stripped before checking (דבאו"ח = דבאורח חיים: the readers carried the prefix into the expansion, the plain checker has no letters for it);
      3. a daf/chapter reference (_reference_fits)."""
    from .detect import parses
    if any(fits(ab, expansion)[0] for _p, ab in parses(surface) or [("", surface)]):
        return True
    ws = expansion.split()
    for n in range(1, 4):
        p = surface[:n]
        if any(c not in STEM_PREFIXES for c in p):
            break
        rest = surface[n:]
        if ws and ws[0].startswith(p) and len(ws[0]) > n and re.search(r'[א-ת]"[א-ת]|\'$', normalize_punct(rest)) and len(norm_letters(rest)) > 1:
            if fits(rest, " ".join([ws[0][n:]] + ws[1:]))[0]:
                return True
    return _reference_fits(surface, expansion)

def _drop_matres(w):
    return w[:1] + w[1:].replace("ו", "").replace("י", "")

def _fits_lenient(letters, ws, trunc):
    al = _align(letters, tuple(ws), lenient=True)
    if al is not None:
        return list(al)
    if trunc and len(ws) == 1:                       # truncated abbreviation: compare with matres removed on both sides
        l2, w2 = _drop_matres(letters), _drop_matres(ws[0])
        if len(l2) >= 2:
            for p in _stems(w2):
                if w2[p:].startswith(l2):
                    return [(ws[0], p, len(l2))]
    return None


def load_irregular(path=None):
    from pathlib import Path
    p = Path(path) if path else Path(__file__).parent / "irregular.tsv"
    IRREGULAR.clear()
    if p.exists():
        for line in p.read_text().splitlines():
            a, _, e = line.partition("\t")
            IRREGULAR.setdefault(a, set()).add(e)

load_irregular()
